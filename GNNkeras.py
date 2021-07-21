from __future__ import annotations

import tensorflow as tf

#######################################################################################################################
### CLASS GNN - NODE BASED ############################################################################################
#######################################################################################################################
class GNNnodeBased(tf.keras.Model):
    def __init__(self,
                 net_state: tf.keras.models.Sequential,
                 net_output: tf.keras.models.Sequential,
                 state_vect_dim: int,
                 max_iteration: int,
                 state_threshold: float) -> None:
        """ CONSTRUCTOR

        :param net_state: (tf.keras.model.Sequential) MLP for the state network, initialized externally.
        :param net_output: (tf.keras.model.Sequential) MLP for the output network, initialized externally.
        :param state_vect_dim: None or (int)>=0, vector dim for a GNN which does not initialize states with node labels.
        :param max_iteration: (int) max number of iteration for the unfolding procedure (to reach convergence).
        :param threshold: (float) threshold for specifying if convergence is reached or not.
        """
        assert state_vect_dim >= 0
        assert max_iteration > 0
        assert state_threshold >= 0

        super().__init__()
        self.net_state = net_state
        self.net_output = net_output
        self.state_vect_dim = int(state_vect_dim)
        self.max_iteration = int(max_iteration)
        self.state_threshold = state_threshold

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self, copy_weights: bool = True) -> GNNnodeBased:
        """ COPY METHOD

        :param copy_weights: (bool) True: state and output weights are copied in new gnn, otherwise they are re-initialized.
        :return: a Deep Copy of the GNN instance.
        """
        # MLPs
        netS = tf.keras.models.clone_model(self.net_state)
        netO = tf.keras.models.clone_model(self.net_output)
        if copy_weights:
            netS.set_weights(self.net_state.get_weights())
            netO.set_weights(self.net_output.get_weights())

        # return copy
        return self.__class__(netS, netO, self.state_vect_dim, self.max_iteration, self.state_threshold)

    ## SAVE AND LOAD METHODs ##########################################################################################
    def save(self, path: str, *args):
        """ Save model to folder <path>, without extra_metrics info """
        from json import dump

        # check path
        if path[-1] != '/': path += '/'

        # save net_state and net_output
        tf.keras.models.save_model(self.net_state, f'{path}net_state/')
        tf.keras.models.save_model(self.net_output, f'{path}net_output/')

        # save configuration file in json format
        config = {'state_vect_dim': self.state_vect_dim,
                  'max_iteration': self.max_iteration,
                  'state_threshold': self.state_threshold}

        with open(f'{path}config.json', 'w') as json_file:
            dump(config, json_file)

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def load(self, path: str):
        """ Load model from folder <path> """
        from json import loads

        # check path
        if path[-1] != '/': path += '/'

        # load configuration file
        with open(f'{path}config.json', 'r') as read_file:
            config = loads(read_file.read())

        # load net_state and net_output
        netS = tf.keras.models.load_model(f'{path}net_state/', compile=False)
        netO = tf.keras.models.load_model(f'{path}net_output/', compile=False)

        return self(net_state=netS, net_output=netO, **config)

    ## COMPILE METHOD #################################################################################################
    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.net_state.compile(*args, **kwargs)
        self.net_output.compile(*args, **kwargs)

    ## CALL METHOD ####################################################################################################
    def call(self, inputs, training: bool = False, mask=None):
        # get a list from :param inputs: tuple, so as to set elements in list (since a tuple is not settable)
        inputs = list(inputs)

        # squeeze inputs: [2] set mask, [3] output mask to make them 1-dimensional (length,)
        inputs[2], inputs[3] = [tf.squeeze(inputs[i], axis=-1) for i in [2, 3]]

        # initialize sparse tensors -> [4] adjacency (nodes, nodes), [5] arcnode (nodes, arcs)
        inputs[4] = tf.SparseTensor(inputs[4][0], values=tf.squeeze(inputs[4][1]), dense_shape=[inputs[0].shape[0], inputs[0].shape[0]])
        inputs[5] = tf.SparseTensor(inputs[5][0], values=tf.squeeze(inputs[5][1]), dense_shape=[inputs[0].shape[0], inputs[1].shape[0]])

        return self.Loop(*inputs, training=training)[-1]

    ## LOOP METHODS ###################################################################################################
    def condition(self, k, state, state_old, *args) -> tf.bool:
        """ Boolean function condition for tf.while_loop correct processing graphs """
        # distance_vector is the Euclidean Distance: √ Σ(xi-yi)² between current state xi and past state yi
        outDistance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(state, state_old)), axis=1))

        # state_norm is the norm of state_old, defined by ||state_old|| = √ Σxi²
        state_norm = tf.sqrt(tf.reduce_sum(tf.square(state_old), axis=1))

        # boolean vector that stores the "convergence reached" flag for each node
        scaled_state_norm = tf.math.scalar_mul(self.state_threshold, state_norm)

        # check whether global convergence and/or the maximum number of iterations have been reached
        checkDistanceVec = tf.greater(outDistance, scaled_state_norm)

        # compute boolean
        c1 = tf.reduce_any(checkDistanceVec)
        c2 = tf.less(k, self.max_iteration)
        return tf.logical_and(c1, c2)

    # -----------------------------------------------------------------------------------------------------------------
    def convergence(self, k, state, state_old, nodes, transposed_adjacency, aggregated_nodes, aggregated_arcs, training) -> tuple:
        """ Compute new state for the graph's nodes """
        # node_components refers to the considered nodes, NOT to the neighbors.
        # It is composed of [nodes' state] if state at t=0 is NOT initialized by labels, [nodes' state | nodes' labels] otherwise
        node_components = [tf.constant(state)] + (nodes if self.state_vect_dim else [])

        # aggregated_states is the aggregation of ONLY neighbors' states.
        # NOTE: if state_vect_dim != 0, neighbors' label are considered using :param aggregated_nodes: since it is constant
        aggregated_states = tf.sparse.sparse_dense_matmul(transposed_adjacency, state)

        # concatenate the destination node 'old' states to the incoming message, to obtain the input to net_state
        inp_state = tf.concat(node_components + [aggregated_states, aggregated_nodes, aggregated_arcs], axis=1)

        # compute new state and update step iteration counter
        state_new = self.net_state(inp_state, training=training)

        return k + 1, state_new, state, nodes, transposed_adjacency, aggregated_nodes, aggregated_arcs, training

    # -----------------------------------------------------------------------------------------------------------------
    def apply_filters(self, state_converged, nodes, adjacency, arcs_label, mask) -> tf.Tensor:
        """ Takes only nodes' [states] or [states|labels] for those with output_mask==1 AND belonging to set """
        if self.state_vect_dim: state_converged = tf.concat([state_converged, nodes], axis=1)
        # return tf.boolean_mask(state_converged, tf.squeeze(mask))
        return tf.boolean_mask(state_converged, mask)

    # -----------------------------------------------------------------------------------------------------------------
    def Loop(self, nodes, arcs, set_mask, output_mask, transposed_adjacency, transposed_arcnode, nodegraph,
             training: bool = False) -> tuple[int, tf.Tensor, tf.Tensor]:
        """ Process a single GraphObject/GraphTensor element g, returning iteration, states and output """

        # get tensorflow dtype
        dtype = tf.keras.backend.floatx()

        # initialize states and iters for convergence loop
        # including aggregated neighbors' label and aggregated incoming arcs' label
        aggregated_arcs = tf.sparse.sparse_dense_matmul(transposed_arcnode, arcs[:, 2:])
        aggregated_nodes = tf.zeros(shape=(nodes.shape[0], 0), dtype=dtype)
        if self.state_vect_dim > 0:
            state = tf.random.normal((nodes.shape[0], self.state_vect_dim), stddev=0.1, dtype=dtype)
            aggregated_nodes = tf.concat([aggregated_nodes, tf.sparse.sparse_dense_matmul(transposed_adjacency, nodes)], axis=1)
        else:
            state = tf.constant(nodes, dtype=dtype)
        k = tf.constant(0, dtype=dtype)
        state_old = tf.ones_like(state, dtype=dtype)
        training = tf.constant(training, dtype=bool)

        # loop until convergence is reached
        k, state, state_old, *_ = tf.while_loop(self.condition, self.convergence,
                                                [k, state, state_old, nodes, transposed_adjacency, aggregated_nodes, aggregated_arcs, training])

        # out_st is the converged state for the filtered nodes, depending on g.set_mask
        mask = tf.logical_and(set_mask, output_mask)
        input_to_net_output = self.apply_filters(state, nodes, transposed_adjacency, arcs[:, 2:], mask)

        # compute the output of the gnn network
        out = self.net_output(input_to_net_output, training=training)
        return k, state, out


#######################################################################################################################
### CLASS GNN - EDGE BASED ############################################################################################
#######################################################################################################################
class GNNedgeBased(GNNnodeBased):
    """ GNN for edge-based problem """

    ## LOOP METHODS ###################################################################################################
    def apply_filters(self, state_converged, nodes, transposed_adjacency, arcs_label, mask) -> tf.Tensor:
        """ Takes only nodes' [states] or [states|labels] for those with output_mask==1 AND belonging to set """
        if self.state_vect_dim: state_converged = tf.concat([state_converged, nodes], axis=1)

        # gather source nodes' and destination nodes' state
        states = tf.gather(state_converged, transposed_adjacency.indices)
        states = tf.reshape(states, shape=(arcs_label.shape[0], 2 * state_converged.shape[1]))
        states = tf.cast(states, tf.keras.backend.floatx())

        # concatenate source and destination states (and labels) to arc labels
        arc_state = tf.concat([states, arcs_label], axis=1)

        # takes only arcs states for those with output_mask==1 AND belonging to the set (in case Dataset == 1 Graph)
        return tf.boolean_mask(arc_state, mask)


#######################################################################################################################
### CLASS GNN - GRAPH BASED ###########################################################################################
#######################################################################################################################
class GNNgraphBased(GNNnodeBased):
    """ GNN for graph-based problem """

    ## LOOP METHODS ###################################################################################################
    def Loop(self, *args, nodegraph, training: bool = False) -> tuple[int, tf.Tensor, tf.Tensor]:
        """ Process a single graph, returning iteration, states and output. Output of graph-based problem is the averaged nodes output """
        k, state_nodes, out_nodes = super().Loop(*args, nodegraph, training=training)
        out_gnn = tf.matmul(nodegraph, out_nodes, transpose_a=True)
        return k, state_nodes, out_gnn


#######################################################################################################################
### CLASS LGNN - GENERAL ##############################################################################################
#######################################################################################################################
'''class KERAS_LGNN(tf.keras.Model):
    def __init__(self,
                 gnns,
                 get_state: bool,
                 get_output: bool,
                 training_mode: str) -> None:

        assert training_mode in ['serial', 'parallel', 'residual']
        assert len(set([type(i) for i in gnns])) == 1

        super().__init__()

        ### LGNNs parameter
        self.GNNS_CLASS = type(gnns[0])

        self.gnns = gnns
        self.LAYERS = len(gnns)

        self.get_state = get_state
        self.get_output = get_output

        # training mode: 'serial', 'parallel', 'residual'
        self.training_mode = training_mode

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self, copy_weights: bool = True) -> KERAS_LGNN:
        """ COPY METHOD

        :param copy_weights: (bool) True: state and output weights of gnns are copied in new lgnn, otherwise they are re-initialized.
        :return: a Deep Copy of the LGNN instance.
        """
        return self.__class__(gnns=[i.copy(copy_weights=copy_weights) for i in self.gnns],
                              get_state=self.get_state, get_output=self.get_output)

    ## SAVE AND LOAD METHODs ##########################################################################################
    def save(self, filepath: str, *args):
        """ Save model to folder <path> """
        from json import dump

        # check paths
        if filepath[-1] != '/': filepath += '/'

        # save GNNs
        for i, gnn in enumerate(self.gnns): gnn.save(f'{filepath}GNN{i}/')

        # save configuration file in json format
        gnn_class = {KERAS_GNNNodeBased: 'n', KERAS_GNNedgeBased: 'a', KERAS_GNNgraphBased: 'g'}[self.GNNS_CLASS]
        config = {'get_state': self.get_state, 'get_output': self.get_output, 'gnns_class': gnn_class, 'training_mode': self.training_mode}

        with open(f'{filepath}config.json', 'w') as json_file:
            dump(config, json_file)

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def load(self, path: str):
        """ Load model from folder <path> """
        from json import loads
        from os import listdir
        from os.path import isdir

        # check paths
        if path[-1] != '/': path += '/'

        # load configuration file
        with open(f'{path}config.json', 'r') as read_file:
            config = loads(read_file.read())

        # load GNNs
        gnn_class = {'n': KERAS_GNNNodeBased, 'a': KERAS_GNNedgeBased, 'g': KERAS_GNNgraphBased}[config.pop('gnns_class')]
        gnns = [gnn_class.load(f'{path}{i}') for i in listdir(path) if isdir(f'{path}{i}')]

        return self(gnns=gnns, **config)

    ## CALL METHOD ####################################################################################################
    def call(self, inputs, training: bool = False, mask = None):
        nodes, arcs, set_mask, output_mask, Adj_indices, Adj_values, ArcNode_indices, ArcNode_values, NodeGraph = inputs
        set_mask, output_mask = tf.squeeze(set_mask), tf.squeeze(output_mask)

        # initialize sparse tensors
        Transposed_Adjacency = tf.SparseTensor(Adj_indices, values=tf.squeeze(Adj_values), dense_shape=[nodes.shape[0], nodes.shape[0]])
        Transposed_ArcNode = tf.SparseTensor(ArcNode_indices, values=tf.squeeze(ArcNode_values),
                                             dense_shape=[nodes.shape[0], arcs.shape[0]])
        return self.Loop(nodes, arcs, set_mask, output_mask, Transposed_Adjacency, Transposed_ArcNode, NodeGraph, training=training)[-1]

    ## LOOP METHODS ###################################################################################################
    def update_graph(self, nodes, arcs, set_mask, output_mask, state, output):
        """
        :param g: (GraphTensor) single GraphTensor element the update process is based on
        :param state: (tensor) output of the net_state model of a single gnn layer
        :param output: (tensor) output of the net_output model of a single gnn layer
        :return: (GraphTensor) a new GraphTensor where actual state and/or output are integrated in nodes/arcs label
        """
        # get tensorflow dtype
        dtype = tf.keras.backend.floatx()

        nodes, arcs = tf.constant(nodes), tf.constant(arcs)

        # define tensors with shape[1]==0 so that it can be concatenate with tf.concat()
        nodeplus = tf.zeros((nodes.shape[0], 0), dtype=dtype)
        arcplus = tf.zeros((arcs.shape[0], 0), dtype=dtype)

        # check state
        if self.get_state: nodeplus = tf.concat([nodeplus, state], axis=1)

        # check output
        if self.get_output:
            # process output to make it shape compatible.
            # Note that what is concatenated is not nodeplus/arcplus, but out, as it has the same length of nodes/arcs
            mask = tf.logical_and(set_mask, output_mask)

            # scatter_nd creates a zeros matrix 'node or arcs-compatible' with the elements of output located in mask==True
            out = tf.scatter_nd(tf.where(mask), output, shape=(len(mask), output.shape[1]))

            if self.GNNS_CLASS == KERAS_GNNedgeBased:
                arcplus = tf.concat([arcplus, out], axis=1)
            else:
                nodeplus = tf.concat([nodeplus, out], axis=1)

        nodes = tf.concat([nodes, nodeplus], axis=1)
        arcs = tf.concat([arcs, arcplus], axis=1)
        return nodes, arcs

    # -----------------------------------------------------------------------------------------------------------------
    def Loop(self, nodes, arcs, set_mask, output_mask, transposed_adjacency, transposed_arcnode, nodegraph,
             training: bool = False) -> tuple[list[tf.Tensor], tf.Tensor, list[tf.Tensor]]:
        """ Process a single GraphObject/GraphTensor element g, returning iteration, states and output """

        constant_inputs = [set_mask, output_mask, transposed_adjacency, transposed_arcnode, nodegraph, training]

        # deep copy of nodes and arcs
        dtype = tf.keras.backend.floatx()
        nodes_0, arcs_0 = tf.constant(nodes, dtype=dtype), tf.constant(arcs, dtype=dtype)

        # forward pass
        K, outs = list(), list()
        for idx, gnn in enumerate(self.gnns[:-1]):

            if isinstance(gnn, KERAS_GNNgraphBased):
                k, state, out = super(KERAS_GNNgraphBased, gnn).Loop(nodes, arcs, *constant_inputs)
                outs.append(tf.matmul(nodegraph, out, transpose_a=True))

            else:
                k, state, out = gnn.Loop(nodes, arcs, *constant_inputs)
                outs.append(out)

            K.append(k)

            # update graph with state and output of the current GNN layer, to feed next GNN layer
            nodes, arcs = self.update_graph(nodes_0, arcs_0, set_mask, output_mask, state, out)

        # final GNN k, state and out values
        k, state, out = self.gnns[-1].Loop(nodes, arcs, *constant_inputs)
        return K + [k], state, outs + [out]


    def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, *args, **kwargs):

        class parallel_loss(tf.keras.losses.Loss):
            def __init__(self, loss):
                super().__init__(name='parallel_loss')
                self.compiled_loss = loss

            def call(self, y_true, y_pred):
                #return tf.reduce_sum(tf.reduce_mean([self.compiled_loss(y_true, y) for y in y_pred]))
                return [self.compiled_loss(y_true, y) for y in y_pred]

        class residual_loss(tf.keras.losses.Loss):
            def __init__(self, loss):
                super().__init__(name='residual_loss')
                self.compiled_loss = loss

            def call(self, y_true, y_pred):
                return tf.reduce_sum(self.compiled_loss(y_true, tf.reduce_mean(y_pred, axis=0)))

        for gnn in self.gnns: gnn.compile(optimizer, loss, metrics, loss_weights, *args, **kwargs)


        from tensorflow.python.keras.engine import compile_utils
        compiled_loss = compile_utils.LossesContainer(loss, loss_weights=loss_weights, output_names=self.output_names)

        if self.training_mode == 'parallel': compiled_loss = parallel_loss(compiled_loss)
        else: compiled_loss = residual_loss(compiled_loss)

        super().compile(optimizer, compiled_loss, metrics, loss_weights, *args, **kwargs)


    ## FIT METHOD #####################################################################################################
    def fit(self, *input, **kwargs):
        if self.training_mode == 'serial':
            for idx, gnn in enumerate(self.gnns):
                print(f'\n\n --- GNN{idx}/{self.LAYERS} ---')
                gnn.fit(*input, **kwargs)
        else: super().fit(*input, **kwargs)
'''
