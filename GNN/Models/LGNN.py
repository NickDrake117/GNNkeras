# codinf=utf-8
from typing import Union

from numpy import savez, load
import tensorflow as tf
from GNN.Models.GNN import GNNnodeBased, GNNarcBased, GNNgraphBased


#######################################################################################################################
### CLASS LGNN - GENERAL ##############################################################################################
#######################################################################################################################
class LGNN(tf.keras.Model):
    """ Layered Graph Neural Network (LGNN) model for node-focused, arc-focused or graph-focused applications. """

    ## CONSTRUCTORS METHODs ###########################################################################################
    def __init__(self,
                 gnns: Union[list[GNNnodeBased], list[GNNarcBased], list[GNNgraphBased]],
                 get_state: bool,
                 get_output: bool) -> None:
        """ CONSTRUCTOR

        :param gnns: list of GNN models belonging to only one of the following types: GNNnodeBased, GNNedgeBased or GNNgraphBased.
        :param get_state: (bool) If True, nodes state is propagated along GNNs layers.
        :param get_output: (bool) If True, nodes/arcs output is propagated along GNNs layers. """

        assert get_state or get_output
        assert len(set([type(i) for i in gnns])) == 1

        super().__init__()

        ### LGNN parameter.
        self.gnns = gnns
        self._get_state = bool(get_state)
        self._get_output = bool(get_output)

        # net_state weights policy: (bool),
        # if True weights are averaged srt the number of iterations, otherwise they're summed.
        self._average_st_grads = None
        # training mode, to be compiled: 'serial', 'parallel', 'residual'.
        self._training_mode = None

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self, copy_weights: bool = True):
        """ COPY METHOD

        :param copy_weights: (bool) True: state and output weights are copied in new gnn, otherwise they are re-initialized.
        :return: a Deep Copy of the LGNN instance. """

        # get configuration dictionary.
        config = self.get_config()
        config["gnns"] = [i.copy(copy_weights=copy_weights) for i in config["gnns"]]

        return self.from_config(config)

    @property
    def get_state(self):
        return self._get_state

    @property
    def get_output(self):
        return self._get_output

    @property
    def training_mode(self):
        return self._training_mode

    @property
    def average_st_grads(self):
        return self._average_st_grads

    @property
    def LAYERS(self):
        return len(self.gnns)

    @property
    def GNN_CLASS(self):
        return type(self.gnns[0])

    @property
    def processing_function(self):
        # get processing function to retrieve state and output for the nodes of the graphs processed by the gnn layer.
        # Fundamental for graph-focused problem, since the output is referred to the entire graph, rather than to the graph nodes.
        # Since in CONSTRUCTOR GNNs type must be only one, type(self.gnns[0]) is exploited
        # and processing function is the same overall GNNs layers.
        return GNNnodeBased.Loop if self.GNN_CLASS._name == "graph" else self.GNN_CLASS.Loop

    ## CONFIG METHODs #################################################################################################
    def get_config(self):
        """ Get configuration dictionary. To be used with from_config().
        It is good practice providing this method to user. """
        return {"gnns": self.gnns, "get_state": self.get_state, "get_output": self.get_output}

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        """ Create class from configuration dictionary. To be used with get_config().
        It is good practice providing this method to user. """
        return cls(**config)

    ## REPR METHODs ###################################################################################################
    def __repr__(self):
        """ Representation string for the instance of LGNN. """
        #return f"LGNN(type={self.__gnnClass__(self.GNN_CLASS)}, layers={self.LAYERS}, " \
        return f"LGNN(type={self.GNN_CLASS._name}, layers={self.LAYERS}, " \
               f"get_state={self.get_state}, get_output={self.get_output}, " \
               f"mode={self.training_mode}, avg={self.average_st_grads})"

    # -----------------------------------------------------------------------------------------------------------------
    def __str__(self):
        """ Representation string for the instance of LGNN, for print() purpose. """
        return self.__repr__()

    ## SAVE AND LOAD METHODs ##########################################################################################
    def save(self, path: str, *args, **kwargs):
        """ Save model to folder <path>.

        :param path: (str) path in which model is saved.
        :param args: args argument of tf.keras.models.save_model function.
        :param kwargs: kwargs argument of tf.keras.models.save_model function. """

        # check paths
        if path[-1] != '/': path += '/'

        # get configuration dictionary.
        config = self.get_config()
        config["gnn_class"] = {'node': 0, 'arc': 1, 'graph': 2}[self.GNN_CLASS._name]

        # save GNNs.
        for i, gnn in enumerate(config.pop("gnns")): gnn.save(f'{path}GNN{i}/', **kwargs)

        # save configuration file in json format.
        savez(f"{path}config.npz", **config)
        from json import dump
        with open(f'{path}config.json', 'w') as json_file:
            dump(config, json_file)

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str):
        """ Load model from folder <path>.
        The new model needs to be compiled to set training_mode and average_st_grads attributes

        :param path: (str) path from which model is loaded.
        :param args: args argument of tf.keras.models.load_model function.
        :param kwargs: kwargs argument of tf.keras.models.load_model function. """

        from json import loads
        from os import listdir
        from os.path import isdir

        # check paths.
        if path[-1] != '/': path += '/'

        # load configuration file.
        config = dict(load(f"{path}config.npz"))

        # load GNNs.
        gnn_class = {0: GNNnodeBased, 1: GNNarcBased, 2: GNNgraphBased}[int(config.pop('gnn_class'))]
        gnns = [gnn_class.load(f'{path}{i}') for i in listdir(path) if isdir(f'{path}{i}')]

        return cls(gnns=gnns, **config)

    ## COMPILE METHOD #################################################################################################
    def compile(self, *args, training_mode: str = 'parallel', average_st_grads: bool = False, **kwargs):
        """ Configures the model for learning.

        :param args: args inherited from Model.compile method. See source for details.
        :param average_st_grads: (bool) If True, net_state params are averaged wrt the number of iterations, summed otherwise.
        :param training_mode: str in ['serial', 'parallel', 'residual'].
                                > serial - each gnn in self.gnns is trained separately, one after another;
                                > parallel - gnns are trained all at once. Loss is processed as mean(loss(t, out_i) for i in self.gnns);
                                > residual - gnns are trained all at once. Loss is processed as loss(t, mean(out_i for i in self.gnns));
        :param average_st_grads: (bool) If True, net_state params are averaged wrt the number of iterations, summed otherwise.
        :param kwargs: Arguments supported for backwards compatibility only. Inherited from Model.compile method. See source for details.
        :raise: ValueError – In case of invalid arguments for `optimizer`, `loss` or `metrics`. """

        # force eager execution, since graph-mode must be implemented.
        run_eagerly = kwargs.pop("run_eagerly", False)

        super().compile(*args, **kwargs, run_eagerly=True)
        for gnn in self.gnns: gnn.compile(*args, average_st_grads=average_st_grads, **kwargs, run_eagerly=run_eagerly)
        self._training_mode = training_mode
        self._average_st_grads = average_st_grads

    ## CALL METHODs ###################################################################################################
    def call(self, inputs, training: bool = False, mask=None, **kwargs):
        """ Call method, get the output of the model for an input graph. Return only output if testing mode.

        :param inputs: (tuple) coming from a GraphSequencer.__getitem__ method, since GNN cannot digest graph as they are.
        :param training: (bool) True/False for training or testing mode, respectively.
        :param mask: inherited from Model.call method. Useless here. Inserted just to avoid warning messages.

        :return: only output of the model if training == False, or a tuple of 3 elements describing, respectively:
        the iteration number reached at the end of Loop method at time T, the nodes state at time T and the output of the model. """
        inputs = self.process_inputs(inputs)
        k, state, out = self.Loop(*inputs, training=training)
        if training: return k, state, out
        return out[kwargs.get('idx', -1)]

    ## STATIC METHODs #################################################################################################
    process_inputs = staticmethod(GNNnodeBased.process_inputs)
    #__gnnClass__ = staticmethod(lambda x: {GNNnodeBased:"node", GNNarcBased: "arc", GNNgraphBased: "graph"}[x])
    __gnnClassLoader__ = staticmethod(lambda x: {"node": GNNnodeBased, "arc": GNNarcBased, "graph": GNNgraphBased}[x])

    ## LOOP METHODS ###################################################################################################
    def update_graph(self, nodes, arcs, dim_node_features, set_mask, output_mask, state, output) -> tuple:
        """ Update nodes and arcs tensor based on get_state and get_output attributes.
        All quantities refer to a single graph, while state and output to GNN state and output calculations.

        :param state: (tensor) output of the net_state model of a single gnn layer.
        :param output: (tensor) output of the net_output model of a single gnn layer.
        :return: (tuple of tensor) new nodes and arcs tensors in which actual state and/or output are integrated.
        """
        # get tensorflow dtype.
        dtype = tf.keras.backend.floatx()

        # get a copy of node and arcs, to prevent strange behaviours.
        # #EastherEgg: it's meglio prevenire that curare.
        nodes, arcs = tf.constant(nodes, dtype=dtype), tf.constant(arcs, dtype=dtype)

        # define tensors with shape[1]==0 so that it can be concatenate with tf.concat().
        nodeplus = tf.zeros((nodes.shape[0], 0), dtype=dtype)
        arcplus = tf.zeros((arcs.shape[0], 0), dtype=dtype)

        # check state.
        if self.get_state: nodeplus = tf.concat([nodeplus, state], axis=1)

        # check output.
        if self.get_output:
            # process output to make it shape compatible.
            # Note that what is concatenated is not nodeplus/arcplus, but out, as it has the same length of nodes/arcs.
            mask = tf.logical_and(set_mask, output_mask)

            # scatter_nd creates a zeros matrix 'node or arcs-compatible' with the elements of output located in mask==True.
            out = tf.scatter_nd(tf.where(mask), output, shape=(len(mask), output.shape[1]))

            if self.GNN_CLASS == GNNarcBased: arcplus = tf.concat([arcplus, out], axis=1)
            else: nodeplus = tf.concat([nodeplus, out], axis=1)

        # update nodes and arcs labels + node label dim.
        nodes = tf.concat([nodeplus, nodes], axis=1)
        arcs = tf.concat([arcplus, arcs], axis=1)
        dim_node_features = dim_node_features + nodeplus.shape[1]
        
        return nodes, arcs, dim_node_features

    # -----------------------------------------------------------------------------------------------------------------
    def Loop(self, nodes, arcs, dim_node_features, set_mask, output_mask, adjacency, arcnode, nodegraph,
             training: bool = False) -> tuple[list[tf.Tensor], tf.Tensor, list[tf.Tensor]]:
        """ Process a single GraphTensor element, returning 3 lists of iterations, states and outputs. """
        constant_inputs = [set_mask, output_mask, adjacency, arcnode, nodegraph]

        # deep copy of nodes and arcs at time t==0.
        dtype = tf.keras.backend.floatx()
        nodes_0, arcs_0 = tf.constant(nodes, dtype=dtype), tf.constant(arcs, dtype=dtype)

        # forward pass.
        K, states, outs = list(), list(), list()
        for idx, gnn in enumerate(self.gnns[:-1]):
            # process graph.
            k, state, out = self.processing_function(gnn, nodes, arcs, dim_node_features, *constant_inputs, training=training)

            # append new k, new states and new gnn output.
            K.append(k)
            states.append(state)
            outs.append(tf.sparse.sparse_dense_matmul(nodegraph, out, adjoint_a=True) if isinstance(gnn, GNNgraphBased) else out)

            # update graph with nodes' state and  nodes/arcs' output of the current GNN layer, to feed next GNN layer.
            nodes, arcs, dim_node_features = self.update_graph(nodes_0, arcs_0, dim_node_features, set_mask, output_mask, state, out)

        # final GNN k, state and out values.
        k, state, out = self.gnns[-1].Loop(nodes, arcs, dim_node_features, *constant_inputs, training=training)

        # return 3 lists of Ks, states and gnn outputs, s.t. len == self.LAYERS.
        return K + [k], states + [state], outs + [out]

    ## FIT METHOD #####################################################################################################
    def train_step(self, data):
        """ training step used for fitting models. """

        # Retrieve data from GraphSequencer.
        x, y, sample_weight = data

        # Run forward pass.
        with tf.GradientTape() as tape:
            k, state, y_pred = self(x, training=True)
            if self.training_mode == 'parallel':
                loss = tf.reduce_mean([self.compiled_loss(y, yi, sample_weight, regularization_losses=self.losses) for yi in y_pred], axis=0)
            else: loss = self.compiled_loss(y, tf.reduce_mean(y_pred, axis=0), sample_weight, regularization_losses=self.losses)

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

        self.optimizer.apply_gradients(zip(dW, W))
        self.compiled_metrics.update_state(y, y_pred[-1], sample_weight)

        # Collect metrics to return.
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict): return_metrics.update(result)
            else: return_metrics[metric.name] = result
        return return_metrics

    # -----------------------------------------------------------------------------------------------------------------
    def fit(self, *input, **kwargs):
        """ Fit method. Re-defined so as to consider the 'serial' learning mode of the LGNN model. """
        if self.training_mode == 'serial':
            ### in serial mode, :param callbacks: must be a list of list/tuple of callbacksObject, s.t. len(callbacks) == self.LAYERS
            # callbacks option.
            callbacks = [list() for _ in range(self.LAYERS)]
            if 'callbacks' in kwargs: callbacks = kwargs.pop('callbacks')

            # training data at t==0.
            training_sequencer_t0 = input[0]
            training_sequencer = training_sequencer_t0.copy()

            # validation data at t==0, if provided.
            validation_sequencer_t0 = kwargs.pop('validation_data', None)
            validation_sequencer = None
            if validation_sequencer_t0 is not None: validation_sequencer = validation_sequencer_t0.copy()

            ### LEARNING PROCEDURE - each gnn layer is trained separately, one after another.
            # for loop only for len(self.gnns)-1, as last layer can be directly trained,
            # without updating nodes/arcs labels after training procedure.
            for idx, gnn in enumerate(self.gnns[:-1]):
                print(f'\n\n --- GNN {idx + 1}/{self.LAYERS} ---')

                ### TRAINING GNN single layer.
                gnn.fit(training_sequencer, *input[1:], **kwargs, callbacks=callbacks[idx], validation_data=validation_sequencer)

                ### PROCESSING TRAINING DATA.
                # set batch size == 1 to retrieve single graph nodes, arcs, state and outputs, for graph update process between layers.
                # shuffle is set to False so that valid_sequence is not shuffled after retrieving sTr and oTr.
                training_sequencer.shuffle = False
                training_sequencer.batch_size = 1

                # retrieve iteration, state and output of nodes/arcs for each graph.
                _, sTr, oTr = zip(*[self.processing_function(gnn, *gnn.process_inputs(inp), training=True)
                                    for inp, *_ in training_sequencer])

                # update nodes and arcs attributes for each single graph.
                training_sequencer = training_sequencer_t0.copy()
                for g, G, s, o in zip(training_sequencer.data, training_sequencer._data, sTr, oTr):
                    n, a, l = self.update_graph(g.nodes, g.arcs, g.DIM_NODE_FEATURES, G.set_mask, G.output_mask, s, o)
                    g.nodes, g.arcs, g._DIM_NODE_FEATURES = n.numpy(), a.numpy(), l
                training_sequencer.build_batches()

                ### PROCESSING VALIDATION DATA if provided, same procedure as the one processing training data.
                if validation_sequencer is not None:
                    # set batch size == 1 to retrieve single graph nodes, arcs, state and outputs, for graph update process between layers.
                    # shuffle is sst to False so that valid_sequence is not shuffled after retrieving sVa and oVa.
                    validation_sequencer.shuffle = False
                    validation_sequencer.batch_size = 1

                    # retrieve iteration, state and output of nodes/arcs for each graph.
                    _, sVa, oVa = zip(*[self.processing_function(gnn, *gnn.process_inputs(inp), training=True)
                                        for inp, *_ in validation_sequencer])

                    # update nodes and arcs attributes for each single graph.
                    validation_sequencer = validation_sequencer_t0.copy()
                    for g, G, s, o in zip(validation_sequencer.data, validation_sequencer._data, sVa, oVa):
                        n, a, l = self.update_graph(g.nodes, g.arcs, g.DIM_NODE_FEATURES, G.set_mask, G.output_mask, s, o)
                        g.nodes, g.arcs, g._DIM_NODE_FEATURES = n.numpy(), a.numpy(), l
                    validation_sequencer.build_batches()

            ### TRAINING GNN single LAST layer (so that all nodes/arcs labels update procedure is not carried out).
            print(f'\n\n --- GNN {self.LAYERS}/{self.LAYERS} ---')
            self.gnns[-1].fit(training_sequencer, *input[1:], **kwargs, callbacks=callbacks[-1], validation_data=validation_sequencer)
        else:
            # fit LGNN keras.Model as usual.
            super().fit(*input, **kwargs)