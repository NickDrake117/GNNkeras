# codinf=utf-8
import tensorflow as tf


#######################################################################################################################
### CLASS COMPOSITE GNN - NODE BASED ##################################################################################
#######################################################################################################################
class CompositeGNNnodeBased(tf.keras.Model):
    """ Composite GNN for node-based problem """

    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self,
                 net_state: list[tf.keras.models.Sequential],
                 net_output: tf.keras.models.Sequential,
                 state_vect_dim: int,
                 max_iteration: int,
                 state_threshold: float) -> None:
        """ CONSTRUCTOR

        :param net_state: (list of tf.keras.model.Sequential) MLPs for the state networks, initialized externally, one for each node type.
        :param net_output: (tf.keras.model.Sequential) MLP for the output network, initialized externally.
        :param state_vect_dim: None or (int)>=0, vector dim for a GNN which does not initialize states with node labels.
        :param max_iteration: (int) max number of iteration for the unfolding procedure (to reach convergence).
        :param threshold: (float) threshold for specifying if convergence is reached or not.
        """
        assert state_vect_dim >= 0
        assert max_iteration > 0
        assert state_threshold >= 0

        super().__init__()

        # GNN parameters
        self.net_state = net_state
        self.net_output = net_output
        self.state_vect_dim = int(state_vect_dim)
        self.max_iteration = int(max_iteration)
        self.state_threshold = state_threshold
        self.average_st_grads = None

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self, copy_weights: bool = True):
        """ COPY METHOD

        :param copy_weights: (bool) True: state and output weights are copied in new gnn, otherwise they are re-initialized.
        :return: a Deep Copy of the GNN instance.
        """
        # MLPs
        netS = [tf.keras.models.clone_model(i) for i in self.net_state]
        netO = tf.keras.models.clone_model(self.net_output)
        if copy_weights:
            for i, j in zip(netS, self.net_state): i.set_weights(j.get_weights())
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
        for i, elem in enumerate(self.net_state): tf.keras.models.save_model(elem, f'{path}net_state_{i}/')
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
        from os import listdir

        # check path
        if path[-1] != '/': path += '/'

        # load configuration file
        with open(f'{path}config.json', 'r') as read_file:
            config = loads(read_file.read())

        # load net_state and net_output
        net_state_dirs = [f'{path}{i}/' for i in listdir(path) if 'net_state' in i]
        netS = [tf.keras.models.load_model(i, compile=False) for i in net_state_dirs]
        netO = tf.keras.models.load_model(f'{path}net_output/', compile=False)

        return self(net_state=netS, net_output=netO, **config)

    ## SUMMARY METHOD #################################################################################################
    def summary(self, *args, **kwargs):
        super().summary(*args, **kwargs)
        for net in self.net_state + [self.net_output]:
            print('\n\n')
            net.summary(*args, **kwargs)

    ## COMPILE METHOD #################################################################################################
    def compile(self, *args, average_st_grads=False, **kwargs):
        """ Configures the model for training.

        :param args: args inherited from Model.compile method. See source for details.
        :param average_st_grads: boolean. If True, net_state params are averaged wrt the number of iterations returned by Loop, summed otherwise.
        :param kwargs: Arguments supported for backwards compatibility only. Inherited from Model.compile method. See source for details

        :raise: ValueError – In case of invalid arguments for `optimizer`, `loss` or `metrics`.
        """
        super().compile(*args, **kwargs)
        for net in self.net_state: net.compile(*args, **kwargs)
        self.net_output.compile(*args, **kwargs)
        self.average_st_grads = average_st_grads

    ## CALL METHOD ####################################################################################################
    def call(self, inputs, training: bool = False, mask=None):
        inputs = self.process_inputs(inputs)
        k, state, out = self.Loop(*inputs, training=training)
        if training: return k, state, out
        else: return out

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def process_inputs(inputs):
        """ convert some inputs in SparseTensor (not handled by default) and squeeze masks for correct computation """

        # get a list from :param inputs: tuple, so as to set elements in list (since a tuple is not settable)
        inputs = list(inputs)

        #for i in inputs: print(type(i))

        # squeeze inputs: [2] dim node labels, [4] set mask, [5] output mask to make them 1-dimensional (length,)
        #inputs[2], inputs[4], inputs[5] = [tf.squeeze(inputs[i], axis=-1) for i in [2, 4, 5]]
        inputs[2:6] = [tf.squeeze(k, axis=-1) for k in inputs[2:6]]

        # initialize sparse tensors -> [6] adjacency (nodes, nodes), [7] composite adjacency list[(nodes, nodes)], [8] arcnode (arcs, nodes)
        #inputs[6] = tf.SparseTensor(inputs[6][0], values=tf.squeeze(inputs[6][1]), dense_shape=[inputs[0].shape[0], inputs[0].shape[0]])
        #inputs[8] = tf.SparseTensor(inputs[8][0], values=tf.squeeze(inputs[8][1]), dense_shape=[inputs[1].shape[0], inputs[0].shape[0]])
        #inputs[7] = [tf.SparseTensor(i, values=tf.squeeze(v, axis=-1), dense_shape=[inputs[0].shape[0], inputs[0].shape[0]]) for i, v in inputs[7]]
        inputs[6]  = [tf.SparseTensor(indices=i, values=tf.squeeze(v, axis=-1), dense_shape=tf.squeeze(s)) for i, v, s in inputs[6]]
        inputs[7:] = [tf.SparseTensor(k[0], values=tf.squeeze(k[1], axis=-1), dense_shape=tf.squeeze(k[2])) for k in inputs[7:]]
        return inputs

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
    def convergence(self, k, state, state_old, nodes, dim_node_labels, type_mask, adjacency, aggregated_component, training) -> tuple:
        """ Compute new state for the graph's nodes """

        # aggregated_states is the aggregation of ONLY neighbors' states.
        aggregated_states = tf.sparse.sparse_dense_matmul(adjacency, state, adjoint_a=True)

        # concatenate the destination node 'old' states to the incoming message, to obtain the input to net_state
        state_new = list()
        for d, m, net in zip(dim_node_labels, type_mask, self.net_state):
            inp_state_i = tf.concat([nodes[:, :d], state, aggregated_states, aggregated_component], axis=1)
            inp_state_i = tf.boolean_mask(inp_state_i, m)

            # compute new state and update step iteration counter
            state_new.append(net(inp_state_i, training=training))

        # reorder state based on nodes' ordering
        state_new = [tf.scatter_nd(tf.where(m), s, (len(m), s.shape[1])) for m, s in zip(type_mask, state_new)]
        state_new = tf.reduce_sum(state_new, axis=0)

        return k + 1, state_new, state, nodes, dim_node_labels, type_mask, adjacency, aggregated_component, training

    # -----------------------------------------------------------------------------------------------------------------
    def apply_filters(self, state_converged, nodes, adjacency, arcs_label, mask) -> tf.Tensor:
        """ Takes only nodes' [states] or [states|labels] for those with output_mask==1 AND belonging to set """
        return tf.boolean_mask(state_converged, mask)

    # -----------------------------------------------------------------------------------------------------------------
    def Loop(self, nodes, arcs, dim_node_labels, type_mask, set_mask, output_mask, composite_adjacencies, adjacency,
            arcnode, nodegraph, training: bool = False) -> tuple[int, tf.Tensor, tf.Tensor]:
        """ Process a single GraphObject/GraphTensor element g, returning iteration, states and output """

        # get tensorflow dtype
        dtype = tf.keras.backend.floatx()

        # initialize states and iters for convergence loop
        # including aggregated neighbors' label and aggregated incoming arcs' label
        aggregated_nodes = [tf.sparse.sparse_dense_matmul(a, nodes[:, :d], adjoint_a=True) for a, d in zip(composite_adjacencies, dim_node_labels)]
        aggregated_arcs = tf.sparse.sparse_dense_matmul(arcnode, arcs[:, 2:], adjoint_a=True)
        aggregated_component = tf.concat(aggregated_nodes + [aggregated_arcs], axis=1)

        # new values for Loop
        k = tf.constant(0, dtype=dtype)
        if self.state_vect_dim > 0: state = tf.random.normal((nodes.shape[0], self.state_vect_dim), stddev=0.1, dtype=dtype)
        else: state = tf.constant(nodes, dtype=dtype)
        state_old = tf.ones_like(state, dtype=dtype)
        training = tf.constant(training, dtype=bool)

        # loop until convergence is reached
        k, state, state_old, *_ = tf.while_loop(self.condition, self.convergence,
                                                [k, state, state_old, nodes, dim_node_labels, type_mask, adjacency, aggregated_component, training])

        # out_st is the converged state for the filtered nodes, depending on g.set_mask
        mask = tf.logical_and(set_mask, output_mask)
        input_to_net_output = self.apply_filters(state, nodes, adjacency, arcs[:, 2:], mask)

        # compute the output of the gnn network
        out = self.net_output(input_to_net_output, training=training)
        return k, state, out

    ## TRAIN METHODS ##################################################################################################
    def train_step(self, data):
        # works only if data is provided by the custom GraphGenerator
        x, y, sample_weight = data

        # Run forward pass.
        with tf.GradientTape() as tape:
            k, state, y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        if self.loss and y is None:
            raise TypeError('Target data is missing. Your model was compiled with `loss` '
                            'argument and so expects targets to be passed in `fit()`.')

        # Run backwards pass.
        #wS, wO = self.net_state.trainable_variables, self.net_output.trainable_variables
        wS, wO = [j for i in self.net_state for j in i.trainable_variables], self.net_output.trainable_variables
        dwbS, dwbO = tape.gradient(loss, [wS, wO])
        if self.average_st_grads: dwbS = [i / k for i in dwbS]

        # self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.optimizer.apply_gradients(zip(dwbS + dwbO, wS + wO))
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

#######################################################################################################################
### CLASS COMPOSITE GNN - EDGE BASED ##################################################################################
#######################################################################################################################
class CompositeGNNedgeBased(CompositeGNNnodeBased):
    """ Composite GNN for edge-based problem """

    ## LOOP METHODS ###################################################################################################
    def apply_filters(self, state_converged, nodes, adjacency, arcs_label, mask) -> tf.Tensor:
        """ Takes only nodes' [states] or [states|labels] for those with output_mask==1 AND belonging to set """

        # gather source nodes' and destination nodes' state
        states = tf.gather(state_converged, adjacency.indices)
        states = tf.reshape(states, shape=(arcs_label.shape[0], 2 * state_converged.shape[1]))
        states = tf.cast(states, tf.keras.backend.floatx())

        # concatenate source and destination states (and labels) to arc labels
        arc_state = tf.concat([states, arcs_label], axis=1)

        # takes only arcs states for those with output_mask==1 AND belonging to the set (in case Dataset == 1 Graph)
        return tf.boolean_mask(arc_state, mask)


#######################################################################################################################
### CLASS COMPOSITE GNN - GRAPH BASED #################################################################################
#######################################################################################################################
class CompositeGNNgraphBased(CompositeGNNnodeBased):
    """ Composite GNN for graph-based problem """

    ## LOOP METHODS ###################################################################################################
    def Loop(self, *args, training: bool = False) -> tuple[int, tf.Tensor, tf.Tensor]:
        """ Process a single graph, returning iteration, states and output. Output of graph-based problem is the averaged nodes output """
        k, state_nodes, out_nodes = super().Loop(*args, training=training)
        out_gnn = tf.sparse.sparse_dense_matmul(args[-1], out_nodes, adjoint_a=True)
        return k, state_nodes, out_gnn