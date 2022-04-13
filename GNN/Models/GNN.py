# codinf=utf-8
import numpy as np
import tensorflow as tf


#######################################################################################################################
### CLASS GNN - NODE BASED ############################################################################################
#######################################################################################################################
class GNNnodeBased(tf.keras.Model):
    """ Graph Neural Network (GNN) model for node-focused applications. """
    _name = "node"

    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self,
                 net_state: tf.keras.models.Sequential,
                 net_output: tf.keras.models.Sequential,
                 state_vect_dim: int,
                 max_iteration: int,
                 state_threshold: float) -> None:
        """ CONSTRUCTOR

        :param net_state: (tf.keras.model.Sequential) MLP for the state network, initialized externally.
        :param net_output: (tf.keras.model.Sequential) MLP for the output network, initialized externally.
        :param state_vect_dim: (int)>=0, dimension for state vectors in GNN where states_t0 != node labels.
        :param max_iteration: (int)>=0 max number of iteration for the unfolding procedure to reach convergence.
        :param state_threshold: (float)>=0 threshold for specifying if convergence is reached or not. """
        assert state_vect_dim >= 0
        assert max_iteration >= 0
        assert state_threshold >= 0

        # GNN + net_state and net_output models
        super().__init__(name=self.name)
        self.net_state = net_state
        self.net_output = net_output

        # GNN parameters.
        self._state_vect_dim = int(state_vect_dim)
        self._max_iteration = int(max_iteration)
        self._state_threshold = float(state_threshold)

        # net_state weights policy: True or False.
        # if True weights are averaged srt the number of iterations, otherwise they're summed
        self._average_st_grads = None

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self, copy_weights: bool = True):
        """ COPY METHOD

        :param copy_weights: (bool) True: state and output weights are copied in new gnn, otherwise they are re-initialized.
        :return: a Deep Copy of the GNN instance. """

        # get configuration dictionary.
        config = self.get_config()

        # MLPs.
        config["net_state"] = tf.keras.models.clone_model(config["net_state"])
        config["net_output"] = tf.keras.models.clone_model(config["net_output"])
        if copy_weights:
            config["net_state"].set_weights(self.net_state.get_weights())
            config["net_output"].set_weights(self.net_output.get_weights())

        # return copy.
        return self.from_config(config)

    ## PROPERTY GETTERS ###############################################################################################
    @property
    def name(self):
        return self._name

    @property
    def state_vect_dim(self):
        return self._state_vect_dim

    @property
    def max_iteration(self):
        return self._max_iteration

    @property
    def state_threshold(self):
        return self._state_threshold

    @property
    def average_st_grads(self):
        return self._average_st_grads

    ## CONFIG METHODs #################################################################################################
    def get_config(self):
        """ Get configuration dictionary. To be used with from_config().
        It is good practice providing this method to user. """
        return {"net_state": self.net_state,
                "net_output": self.net_output,
                "state_vect_dim": self.state_vect_dim,
                "max_iteration": self.max_iteration,
                "state_threshold": self.state_threshold}

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        """ Create class from configuration dictionary. To be used with get_config().
        It is good practice providing this method to user. """
        return cls(**config)

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self) -> str:
        """ Representation string for the instance of GNN. """
        return f"GNN(type={self.name}, state_dim={self.state_vect_dim}, " \
               f"threshold={self.state_threshold}, max_iter={self.max_iteration}, " \
               f"avg={self.average_st_grads})"

    # -----------------------------------------------------------------------------------------------------------------
    def __str__(self) -> str:
        """ Representation string for the instance of GNN, for print() purpose. """
        return self.__repr__()


    ## SAVE AND LOAD METHODs ##########################################################################################
    def save(self, path: str, *args, **kwargs) -> None:
        """ Save model to folder <path>.

        :param path: (str) path in which model is saved.
        :param args: args argument of tf.keras.models.save_model function.
        :param kwargs: kwargs argument of tf.keras.models.save_model function. """

        # check path.
        if path[-1] != '/': path += '/'

        # get configuration dictionary.
        config = self.get_config()

        # save net_state and net_output.
        tf.keras.models.save_model(config.pop("net_state"), f'{path}net_state/', *args, **kwargs)
        tf.keras.models.save_model(config.pop("net_output"), f'{path}net_output/', *args, **kwargs)

        # save configuration file (without MLPs info) in json format.
        np.savez(f"{path}config.npz", **config)

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str, *args, **kwargs):
        """ Load model from folder <path>.

        :param path: (str) path from which model is loaded.
        :param args: args argument of tf.keras.models.load_model function.
        :param kwargs: kwargs argument of tf.keras.models.load_model function. """

        # check path.
        if path[-1] != '/': path += '/'

        # load net_state and net_output.
        netS = tf.keras.models.load_model(f'{path}net_state/', compile=False, *args, **kwargs)
        netO = tf.keras.models.load_model(f'{path}net_output/', compile=False, *args, **kwargs)

        # load configuration file.
        config = np.load(f"{path}config.npz")
        return cls(net_state=netS, net_output=netO, **config)

    ## SUMMARY METHOD #################################################################################################
    def summary(self, *args, **kwargs):
        """ Summary method, to have a graphical representation for the GNN model. """
        super().summary(*args, **kwargs)
        for net in [self.net_state, self.net_output]:
            print('\n\n')
            net.summary(*args, **kwargs)

    ## COMPILE METHOD #################################################################################################
    def compile(self, *args, average_st_grads=False, **kwargs):
        """ Configures the model for learning.

        :param args: args inherited from Model.compile method. See source for details.
        :param average_st_grads: (bool) If True, net_state params are averaged wrt the number of iterations, summed otherwise.
        :param kwargs: Arguments supported for backwards compatibility only. Inherited from Model.compile method. See source for details.
        :raise: ValueError – In case of invalid arguments for 'optimizer', 'loss' or 'metrics'. """

        # force eager execution on super() model, since graph-mode must be implemented.
        run_eagerly = kwargs.pop("run_eagerly", False)

        super().compile(*args, **kwargs, run_eagerly=True)
        self.net_state.compile(*args, **kwargs, run_eagerly=run_eagerly)
        self.net_output.compile(*args, **kwargs, run_eagerly=run_eagerly)
        self._average_st_grads = average_st_grads

    ## CALL METHODs ###################################################################################################
    def call(self, inputs, training: bool = False, mask=None):
        """ Call method, get the output of the model for an input graph. Return only output if testing mode.

        :param inputs: (tuple) coming from a GraphSequencer.__getitem__ method, since GNN cannot digest graph as they are.
        :param training: (bool) True/False for training or testing mode, respectively.
        :param mask: inherited from Model.call method. Useless here. Inserted just to avoid warning messages.

        :return: only output of the model if training == False, or a tuple of 3 elements describing, respectively:
        the iteration number reached at the end of Loop method at time T, the nodes state at time T and the output of the model. """
        inputs = self.process_inputs(inputs)
        k, state, out = self.Loop(*inputs, training=training)
        if training: return k, state, out
        else: return out

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def process_inputs(inputs):
        """ Convert some inputs in SparseTensor (not handled by default) and squeeze masks for correct computation. """

        # get a list from :param inputs: tuple, so as to set elements in list (since a tuple is not settable).
        inputs = list(inputs)

        # squeeze inputs: [2] dim_node_features, [3] set mask, [4] output mask,
        # to make them 1-dimensional (length,).
        inputs[2:5] = [tf.squeeze(k, axis=-1) for k in inputs[2:5]]

        # initialize sparse tensors -> [5] adjacency, [6] arcnode, [7] nodegraph matrices.
        inputs[5:] = [tf.SparseTensor(k[0], values=tf.squeeze(k[1], axis=-1), dense_shape=tf.squeeze(k[2])) for k in inputs[5:]]
        return inputs

    ## LOOP METHODS ###################################################################################################
    def condition(self, k, state, state_old, *args) -> tf.bool:
        """ Boolean function condition for tf.while_loop correct processing graphs. """

        # distance_vector is the Euclidean Distance: √ Σ(xi-yi)² between current state xi and past state yi.
        outDistance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(state, state_old)), axis=1))

        # state_norm is the norm of state_old, defined by ||state_old|| = √ Σxi².
        state_norm = tf.sqrt(tf.reduce_sum(tf.square(state_old), axis=1))

        # boolean vector that stores the "convergence reached" flag for each node.
        scaled_state_norm = tf.math.scalar_mul(self.state_threshold, state_norm)

        # check whether global convergence and/or the maximum number of iterations have been reached.
        checkDistanceVec = tf.greater(outDistance, scaled_state_norm)

        # compute boolean.
        c1 = tf.reduce_any(checkDistanceVec)
        c2 = tf.less(k, self.max_iteration)
        return tf.logical_and(c1, c2)

    # -----------------------------------------------------------------------------------------------------------------
    def convergence(self, k, state, state_old, nodes, adjacency, aggregated_component, training) -> tuple:
        #aggregated_nodes, aggregated_arcs, training) -> tuple:
        """ Compute new state for the graph's nodes. """

        # node_components refers to the considered nodes, NOT to the neighbors.
        # It is composed of [nodes' state] if state at t=0 is NOT initialized by labels, [nodes' state | nodes' labels] otherwise.
        node_components = [tf.constant(state)]
        if self.state_vect_dim > 0: node_components += [nodes]

        # aggregated_states is the aggregation of ONLY neighbors' states.
        # NOTE: if state_vect_dim != 0, neighbors' label are considered using :param aggregated_nodes: since it is
        # the same values over iterations.
        aggregated_states = tf.sparse.sparse_dense_matmul(adjacency, state, adjoint_a=True)

        # concatenate the destination node 'old' states to the incoming message, to obtain the input to net_state.
        inp_state = tf.concat(node_components + [aggregated_states, aggregated_component], axis=1)

        # compute new state and update step iteration counter.
        state_new = self.net_state(inp_state, training=training)

        return k + 1, state_new, state, nodes, adjacency, aggregated_component, training

    # -----------------------------------------------------------------------------------------------------------------
    def apply_filters(self, state_converged, nodes, adjacency, arcs, mask) -> tf.Tensor:
        """ Takes only nodes' [states] or [states|labels] for those with output_mask==1 AND belonging to set. """
        if self.state_vect_dim: state_converged = tf.concat([state_converged, nodes], axis=1)
        return tf.boolean_mask(state_converged, mask)

    # -----------------------------------------------------------------------------------------------------------------
    def use_net_output(self, x, nodegraph, training : bool = False):
        return self.net_output(x, training=training)

    # -----------------------------------------------------------------------------------------------------------------
    def Loop(self, nodes, arcs, dim_node_features, set_mask, output_mask, adjacency, arcnode, nodegraph,
             training: bool = False) -> tuple[int, tf.Tensor, tf.Tensor]:
        """ Process a single GraphTensor element, returning iteration, states and output. """

        # get tensorflow dtype.
        dtype = tf.keras.backend.floatx()

        # initialize states and iters for convergence loop,
        # including aggregated neighbors' label and aggregated incoming arcs' label.
        aggregated_arcs = tf.sparse.sparse_dense_matmul(arcnode, arcs, adjoint_a=True)
        aggregated_nodes = tf.zeros(shape=(nodes.shape[0], 0), dtype=dtype)
        if self.state_vect_dim > 0:
            state = tf.random.normal((nodes.shape[0], self.state_vect_dim), stddev=0.1, dtype=dtype)
            aggregated_nodes = tf.concat([aggregated_nodes, tf.sparse.sparse_dense_matmul(adjacency, nodes, adjoint_a=True)], axis=1)
        else: state = tf.constant(nodes, dtype=dtype)
        aggregated_component = tf.concat([aggregated_nodes, aggregated_arcs], axis=1)

        # new values for Loop.
        k = tf.constant(0, dtype=dtype)
        state_old = tf.ones_like(state, dtype=dtype)
        training = tf.constant(training, dtype=bool)

        # loop until convergence is reached.
        k, state, state_old, *_ = tf.while_loop(self.condition, self.convergence,
                                                [k, state, state_old, nodes, adjacency, aggregated_component, training])
                                                #[k, state, state_old, nodes, adjacency, aggregated_nodes, aggregated_arcs, training])

        # out_st is the converged state for the filtered nodes, depending on g.set_mask.
        mask = tf.logical_and(set_mask, output_mask)
        input_to_net_output = self.apply_filters(state, nodes, adjacency, arcs, mask)

        # compute the output of the gnn network.
        # out = self.net_output(input_to_net_output, training=training)
        out = self.use_net_output(input_to_net_output, nodegraph, training=training)
        return k, state, out

    ## LEARNING METHODs ###############################################################################################
    def train_step(self, data):
        """ training step used for fitting models. """

        # Retrieve data from GraphSequencer.
        x, y, sample_weight = data

        # Run forward pass.
        with tf.GradientTape() as tape:
            k, state, y_pred = self.call(x, training=True) #from self(....) to self.call(...)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        if self.loss and y is None:
            raise TypeError('Target data is missing. Your model was compiled with `loss` '
                            'argument and so expects targets to be passed in `fit()`.')

        # Run backwards pass and calculate delta_w and delta_b in net_state based on self.average_st_grads
        wS, wO = self.net_state.trainable_variables, self.net_output.trainable_variables
        dwbS, dwbO = tape.gradient(loss, [wS, wO])
        if self.average_st_grads: dwbS = [i / k for i in dwbS]

        self.optimizer.apply_gradients(zip(dwbS + dwbO, wS + wO))
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        # Collect metrics to return.
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict): return_metrics.update(result)
            else: return_metrics[metric.name] = result
        return return_metrics


#######################################################################################################################
### CLASS GNN - EDGE BASED ############################################################################################
#######################################################################################################################
class GNNarcBased(GNNnodeBased):
    """ Graph Neural Network (GNN) model for arc-focused applications. """
    _name = "arc"

    ## LOOP METHODS ###################################################################################################
    def apply_filters(self, state_converged, nodes,  adjacency, arcs, mask) -> tf.Tensor:
        """ Takes only nodes' [states] or [states|labels] for those with output_mask==1 AND belonging to set. """
        if self.state_vect_dim: state_converged = tf.concat([state_converged, nodes], axis=1)

        # gather source nodes' and destination nodes' state.
        states = tf.gather(state_converged, adjacency.indices)
        states = tf.reshape(states, shape=(arcs.shape[0], 2 * state_converged.shape[1]))
        states = tf.cast(states, tf.keras.backend.floatx())

        # concatenate source and destination states (and labels) to arc labels.
        arc_state = tf.concat([states, arcs], axis=1)

        # takes only arcs states for those with output_mask==1 AND belonging to the set (in case Dataset == 1 Graph).
        return tf.boolean_mask(arc_state, mask)


#######################################################################################################################
### CLASS GNN - GRAPH BASED ###########################################################################################
#######################################################################################################################
class GNNgraphBased(GNNnodeBased):
    """ Graph Neural Network (GNN) model for graph-focused applications. """
    _name = "graph"

    # -----------------------------------------------------------------------------------------------------------------
    def use_net_output(self, x, nodegraph, training : bool = False):
        for l in self.net_output.layers[:-1]: x = l(x, training=training)
        return self.net_output.layers[-1](tf.sparse.sparse_dense_matmul(nodegraph, x, adjoint_a=True), training=training)