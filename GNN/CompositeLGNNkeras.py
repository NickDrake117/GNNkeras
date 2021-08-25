from typing import Union

import tensorflow
import tensorflow as tf

from GNN.CompositeGNNkeras import CompositeGNNnodeBased, CompositeGNNedgeBased, CompositeGNNgraphBased


#######################################################################################################################
### CLASS LGNN - GENERAL ##############################################################################################
######################################################################################################################
class CompositeLGNN(tf.keras.Model):
    """ LGNN for general purpose problem """

    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self,
                 gnns: Union[list[CompositeGNNnodeBased], list[CompositeGNNedgeBased], list[CompositeGNNgraphBased]],
                 get_state: bool,
                 get_output: bool) -> None:
        """ CONSTRUCTOR

        :param gnns: list of GNN models belonging to only one of the following types: GNNnodeBased, GNNedgeBased or GNNgraphBased.
        :param get_state: boolean. If True, nodes state is propagated along GNNs layers.
        :param get_output:  boolean. If True, nodes/arcs output is propagated along GNNs layers.
        """
        assert len(set([type(i) for i in gnns])) == 1

        super().__init__()

        ### LGNN parameter
        self.GNNS_CLASS = type(gnns[0])

        self.gnns = gnns
        self.LAYERS = len(gnns)

        self.get_state = get_state
        self.get_output = get_output

        # training mode, to be compiled: 'serial', 'parallel', 'residual'
        self.training_mode = None

        # net_state weights policy: True or False.
        # if True weights are averaged srt the number of iterations, otherwise they're summed
        self.average_st_grads = None

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self, copy_weights: bool = True):
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
        gnn_class = {CompositeGNNnodeBased: 'n', CompositeGNNedgeBased: 'a', CompositeGNNgraphBased: 'g'}[self.GNNS_CLASS]
        config = {'get_state': self.get_state, 'get_output': self.get_output, 'gnns_class': gnn_class}

        with open(f'{filepath}config.json', 'w') as json_file:
            dump(config, json_file)

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def load(self, path: str):
        """ Load model from folder <path>. The new model needs to be compiled to set training_mode and average_st_grads attributes """
        from json import loads
        from os import listdir
        from os.path import isdir

        # check paths
        if path[-1] != '/': path += '/'

        # load configuration file
        with open(f'{path}config.json', 'r') as read_file:
            config = loads(read_file.read())

        # load GNNs
        gnn_class = {'n': CompositeGNNnodeBased, 'a': CompositeGNNedgeBased, 'g': CompositeGNNgraphBased}[config.pop('gnns_class')]
        gnns = [gnn_class.load(f'{path}{i}') for i in listdir(path) if isdir(f'{path}{i}')]

        return self(gnns=gnns, **config)

    ## COMPILE METHOD #################################################################################################
    def compile(self, *args, training_mode: str = 'parallel', average_st_grads: bool = False, **kwargs):
        """ Configures the model for training.

        :param args: args inherited from Model.compile method. See source for details.
        :param training_mode: str in ['serial', 'parallel', 'residual'].
                                > serial - each gnn in self.gnns is trained separately, one after another
                                > parallel - gnns are trained all at once. Loss is processed as mean(loss(t, out_i) for i in self.gnns)
                                > residual - gnns are trained all at once. Loss is processed as loss(t, mean(out_i for i in self.gnns))
        :param average_st_grads: boolean. If True, net_state params are averaged wrt the number of iterations returned by Loop, summed otherwise.
        :param kwargs: Arguments supported for backwards compatibility only. Inherited from Model.compile method. See source for details

        :raise: ValueError – In case of invalid arguments for `optimizer`, `loss` or `metrics`.
        """
        super().compile(*args, **kwargs)
        for gnn in self.gnns: gnn.compile(*args, average_st_grads=average_st_grads, **kwargs)

        self.training_mode = training_mode
        self.average_st_grads = average_st_grads

    ## CALL METHODs ###################################################################################################
    def call(self, inputs, training: bool = False, mask=None):
        inputs = self.process_inputs(inputs)
        k, state, out = self.Loop(*inputs, training=training)
        if training: return k, state, out
        return out[-1]

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def process_inputs(inputs):
        """ convert some inputs in SparseTensor (not handled by default) and squeeze masks for correct computation """

        # get a list from :param inputs: tuple, so as to set elements in list (since a tuple is not settable)
        inputs = list(inputs)

        # squeeze inputs: [2] dim node labels, [4] set mask, [5] output mask to make them 1-dimensional (length,)
        inputs[2], inputs[4], inputs[5] = [tf.squeeze(inputs[i], axis=-1) for i in [2, 4, 5]]

        # initialize sparse tensors -> [6] adjacency (nodes, nodes), [7] composite adjacency list[(nodes, nodes)], [8] arcnode (nodes, arcs)
        inputs[6] = tf.SparseTensor(inputs[6][0], values=tf.squeeze(inputs[6][1]), dense_shape=[inputs[0].shape[0], inputs[0].shape[0]])
        inputs[8] = tf.SparseTensor(inputs[8][0], values=tf.squeeze(inputs[8][1]), dense_shape=[inputs[0].shape[0], inputs[1].shape[0]])
        inputs[7] = [tf.SparseTensor(i, values=tf.squeeze(v, axis=-1), dense_shape=[inputs[0].shape[0], inputs[0].shape[0]]) for i, v in inputs[7]]

        return inputs

    ## LOOP METHODS ###################################################################################################
    def update_graph(self, nodes, arcs, dim_node_labels, set_mask, output_mask, state, output) -> tuple:
        """ update nodes and arcs tensor based on get_state and get_output attributes

        :param state: (tensor) output of the net_state model of a single gnn layer
        :param output: (tensor) output of the net_output model of a single gnn layer
        :return: (tuple of tensor) new nodes and arcs tensors in which actual state and/or output are integrated
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

            if self.GNNS_CLASS == CompositeGNNedgeBased: arcplus = tf.concat([arcplus, out], axis=1)
            else: nodeplus = tf.concat([nodeplus, out], axis=1)

        # update nodes and arcs labels
        nodes = tf.concat([nodeplus, nodes], axis=1)
        arcs = tf.concat([arcplus, arcs], axis=1)
        dim_node_labels = dim_node_labels + nodeplus.shape[1]
        return nodes, arcs, dim_node_labels

    # -----------------------------------------------------------------------------------------------------------------
    def Loop(self, nodes, arcs, dim_node_labels, type_mask, set_mask, output_mask, transposed_adjacency, transposed_composite_adjacencies,
        transposed_arcnode, nodegraph, training: bool = False) -> tuple[list[tf.Tensor], tf.Tensor, list[tf.Tensor]]:

        """ Process a single GraphObject/GraphTensor element g, returning 3 lists of iteration(s), state(s) and output(s) """

        #constant_inputs = [dim_node_labels, type_mask, set_mask, output_mask, transposed_adjacency, transposed_composite_adjacencies, transposed_arcnode, nodegraph]
        constant_inputs = [type_mask, set_mask, output_mask, transposed_adjacency, transposed_composite_adjacencies,
                           transposed_arcnode, nodegraph]

        # deep copy of nodes and arcs
        dtype = tf.keras.backend.floatx()
        nodes_0, arcs_0 = tf.constant(nodes, dtype=dtype), tf.constant(arcs, dtype=dtype)

        # forward pass
        K, states, outs = list(), list(), list()
        for idx, gnn in enumerate(self.gnns[:-1]):
            # process graph
            processing_function = super(CompositeGNNgraphBased, gnn).Loop if isinstance(gnn, CompositeGNNgraphBased) else gnn.Loop
            k, state, out = processing_function(nodes, arcs, dim_node_labels, *constant_inputs, training=training)

            # append new k, new states and new gnn output
            K.append(k)
            states.append(state)
            outs.append(tf.matmul(nodegraph, out, transpose_a=True) if isinstance(gnn, CompositeGNNgraphBased) else out)

            # update graph with nodes' state and  nodes/arcs' output of the current GNN layer, to feed next GNN layer
            nodes, arcs, dim_node_labels = self.update_graph(nodes_0, arcs_0, dim_node_labels, set_mask, output_mask, state, out)

        # final GNN k, state and out values
        k, state, out = self.gnns[-1].Loop(nodes, arcs, dim_node_labels, *constant_inputs, training=training)

        # return a list of Ks, states and gnn outputs, s.t. len == self.LAYERS
        return K + [k], states + [state], outs + [out]

    ## FIT METHOD #####################################################################################################
    def train_step(self, data):
        # works only if data is provided by the custom GraphGenerator
        x, y, sample_weight = data

        # Run forward pass.
        with tf.GradientTape() as tape:
            k, state, y_pred = self(x, training=True)
            if self.training_mode == 'parallel':
                loss = tf.reduce_mean([self.compiled_loss(y, yi, sample_weight, regularization_losses=self.losses) for yi in y_pred], axis=0)
            else:
                loss = self.compiled_loss(y, tf.reduce_mean(y_pred, axis=0), sample_weight, regularization_losses=self.losses)

        if self.loss and y is None:
            raise TypeError('Target data is missing. Your model was compiled with `loss` '
                            'argument and so expects targets to be passed in `fit()`.')

        # Run backwards pass.
        wS, wO = [gnn.net_state.trainable_variables for gnn in self.gnns], [gnn.net_output.trainable_variables for gnn in self.gnns]
        dwbS, dwbO = tape.gradient(loss, [wS, wO])
        if self.average_st_grads: dwbS = [[elem / it for elem in layer] for it, layer in zip(k, dwbS)]

        dW = [i for j in dwbS + dwbO for i in j]
        W = [i for j in wS + wO for i in j]
        assert len(dW) == len(W)

        # self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.optimizer.apply_gradients(zip(dW, W))
        self.compiled_metrics.update_state(y, y_pred[-1], sample_weight)

        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    # -----------------------------------------------------------------------------------------------------------------
    def fit(self, *input, **kwargs):

        if self.training_mode == 'serial':
            ### in serial mode, :param callbacks: must be a list of list/tuple of callbacksOnject, s.t. len(callbacks) == self.LAYERS

            # callbacks option
            callbacks = [list() for _ in range(self.LAYERS)]
            if 'callbacks' in kwargs:
                assert all(isinstance(x, (list, tuple)) for x in kwargs['callbacks'])
                assert len(kwargs['callbacks']) == self.LAYERS
                callbacks = kwargs.pop('callbacks')

            # training data at t==0
            training_data_t0 = input[0]
            gTr_generator = training_data_t0.copy()

            # validation data at t==0, if provided
            validation_data, gVa_generator = None, None
            if 'validation_data' in kwargs:
                validation_data = kwargs.pop('validation_data')
                gVa_generator = validation_data.copy()

            ### LEARNING PROCEDURE - each gnn layer is trained separately, one after another
            for idx, gnn in enumerate(self.gnns):
                print(f'\n\n --- GNN {idx}/{self.LAYERS} ---')

                ### TRAINING GNN single layer
                gnn.fit(gTr_generator.copy(), *input[1:], **kwargs, callbacks=callbacks[idx],
                        validation_data=gVa_generator.copy() if gVa_generator else None)

                # get processing function to retrieve state and output for the nodes of the graphs processed by the gnn layer.
                # It's fundamental for graph-based problem, since the output is referred to the entire graph, rather than to the graph nodes.
                processing_function = super(CompositeGNNgraphBased, gnn).Loop if isinstance(gnn, CompositeGNNgraphBased) else gnn.Loop

                ### PROCESSING TRAINING DATA
                # set batch size == 1 to retrieve single graph nodes, arcs, state and outputs, for graph update process between layers
                gTr_generator.shuffle = False
                gTr_generator.set_batch_size(1)

                # retrieve iteration, state and output of nodes/arcs for each graph
                _, sTr, oTr = zip(*[processing_function(*gnn.process_inputs(i[0]), training=True) for i in gTr_generator])

                # update nodes and arcs attributes for each single graph.
                # Note that graph.DIM_NODE_LABEL is not updated, as the new graph is computed and processed on the fly
                gTr_generator = training_data_t0.copy()
                for g, s, o in zip(gTr_generator.data, sTr, oTr):
                    n, a, l = self.update_graph(g.nodes, g.arcs, g.DIM_NODE_LABEL, g.set_mask, g.output_mask, s, o)
                    g.nodes, g.arcs, g.DIM_NODE_LABEL = n.numpy(), a.numpy(), l

                ### PROCESSING VALIDATION DATA if provided, same as the training data
                if validation_data:
                    # set batch size == 1 to retrieve single graph nodes, arcs, state and outputs, for graph update process between layers
                    gVa_generator.shuffle = False
                    gVa_generator.set_batch_size(1)

                    # retrieve iteration, state and output of nodes/arcs for each graph
                    _, sVa, oVa = zip(*[processing_function(*gnn.process_inputs(i[0]), training=True) for i in gVa_generator])

                    # update nodes and arcs attributes for each single graph.
                    # Note that graph.DIM_NODE_LABEL is not updated, as the new graph is computed and processed on the fly
                    gVa_generator = validation_data.copy()
                    for g, s, o in zip(gVa_generator.data, sVa, oVa):
                        n, a, l = self.update_graph(g.nodes, g.arcs, g.DIM_NODE_LABEL, g.set_mask, g.output_mask, s, o)
                        g.nodes, g.arcs, g.DIM_NODE_LABEL = n.numpy(), a.numpy(), l

        else:
            super().fit(*input, **kwargs)