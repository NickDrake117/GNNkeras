from __future__ import annotations

import tensorflow as tf
from GNNkeras import GNNnodeBased, GNNedgeBased, GNNgraphBased

#######################################################################################################################
### CLASS LGNN - GENERAL ##############################################################################################
#######################################################################################################################
class LGNN(tf.keras.Model):
    def __init__(self,
                 gnns,
                 get_state: bool,
                 get_output: bool) -> None:

        assert len(set([type(i) for i in gnns])) == 1

        super().__init__()

        ### LGNNs parameter
        self.GNNS_CLASS = type(gnns[0])

        self.gnns = gnns
        self.LAYERS = len(gnns)

        self.get_state = get_state
        self.get_output = get_output

        # training mode, to be compiled: 'serial', 'parallel', 'residual'
        self.training_mode = None
        self.average_st_grads = None

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self, copy_weights: bool = True) -> LGNN:
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
        gnn_class = {GNNnodeBased: 'n', GNNedgeBased: 'a', GNNgraphBased: 'g'}[self.GNNS_CLASS]
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
        gnn_class = {'n': GNNnodeBased, 'a': GNNedgeBased, 'g': GNNgraphBased}[config.pop('gnns_class')]
        gnns = [gnn_class.load(f'{path}{i}') for i in listdir(path) if isdir(f'{path}{i}')]
        training_mode = config.pop('training_mode')

        return self(gnns=gnns, **config)

        ## CALL METHOD ####################################################################################################

    def call(self, inputs, training: bool = False, mask=None):
        # get a list from :param inputs: tuple, so as to set elements in list (since a tuple is not settable)
        inputs = list(inputs)

        # squeeze inputs: [2] set mask, [3] output mask to make them 1-dimensional (length,)
        inputs[2], inputs[3] = [tf.squeeze(inputs[i], axis=-1) for i in [2, 3]]

        # initialize sparse tensors -> [4] adjacency (nodes, nodes), [5] arcnode (nodes, arcs)
        inputs[4] = tf.SparseTensor(inputs[4][0], values=tf.squeeze(inputs[4][1]), dense_shape=[inputs[0].shape[0], inputs[0].shape[0]])
        inputs[5] = tf.SparseTensor(inputs[5][0], values=tf.squeeze(inputs[5][1]), dense_shape=[inputs[0].shape[0], inputs[1].shape[0]])

        # return self.Loop(*inputs, training=training)[-1]
        k, state, out = self.Loop(*inputs, training=training)
        if training:
            return k, state, out
        else:
            return out[-1]

    ## LOOP METHODS ###################################################################################################
    def update_graph(self, nodes, arcs, set_mask, output_mask, state, output):
        """
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

            if self.GNNS_CLASS == GNNedgeBased:
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

        constant_inputs = [set_mask, output_mask, transposed_adjacency, transposed_arcnode, nodegraph]

        # deep copy of nodes and arcs
        dtype = tf.keras.backend.floatx()
        nodes_0, arcs_0 = tf.constant(nodes, dtype=dtype), tf.constant(arcs, dtype=dtype)

        # forward pass
        K, outs = list(), list()
        for idx, gnn in enumerate(self.gnns[:-1]):

            if isinstance(gnn, GNNgraphBased):
                k, state, out = super(GNNgraphBased, gnn).Loop(nodes, arcs, *constant_inputs, training=training)
                outs.append(tf.matmul(nodegraph, out, transpose_a=True))

            else:
                k, state, out = gnn.Loop(nodes, arcs, *constant_inputs, training=training)
                outs.append(out)

            K.append(k)

            # update graph with state and output of the current GNN layer, to feed next GNN layer
            nodes, arcs = self.update_graph(nodes_0, arcs_0, set_mask, output_mask, state, out)

        # final GNN k, state and out values
        k, state, out = self.gnns[-1].Loop(nodes, arcs, *constant_inputs, training=training)
        return K + [k], state, outs + [out]

    # -----------------------------------------------------------------------------------------------------------------
    def compile(self, *args, training_mode ='parallel', average_st_grads=False, **kwargs):
        super().compile(*args, **kwargs)
        for gnn in self.gnns: gnn.compile(*args, average_st_grads=average_st_grads, **kwargs)

        self.training_mode = training_mode
        self.average_st_grads = average_st_grads


    ## FIT METHOD #####################################################################################################
    def fit(self, *input, **kwargs):
        #if self.training_mode is None: self.training_mode = 'parallel'

        if self.training_mode == 'serial':
            for idx, gnn in enumerate(self.gnns):
                print(f'\n\n --- GNN{idx}/{self.LAYERS} ---')
                gnn.fit(*input, **kwargs)
        else: super().fit(*input, **kwargs)

    def train_step(self, data):
        # works only if data is provided by the custom GraphGenerator
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

