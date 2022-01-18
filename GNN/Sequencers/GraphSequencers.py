# codinf=utf-8
import numpy as np
import tensorflow as tf

from GNN.composite_graph_class import CompositeGraphObject, CompositeGraphTensor
from GNN.graph_class import GraphObject, GraphTensor


#######################################################################################################################
### CLASS GRAPH SEQUENCER FOR MULTIPLE HOMOGENEOUS DATA ### FOR FEEDING THE MODEL DURING LEARNING PROCESS ### BASE ####
#######################################################################################################################
class MultiGraphSequencer(tf.keras.utils.Sequence):
    """ GraphSequencer for dataset composed of multiple Homogeneous Graphs. """

    # Specific function utilities
    merge = classmethod(GraphObject.merge)
    to_graph_tensor = classmethod(GraphTensor.fromGraphObject)

    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self,
                 graphs: list[GraphObject],
                 focus: str,
                 aggregation_mode: str,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """ CONSTRUCTOR

        :param graphs: a list of GraphObject elements to be sequenced.
        :param focus: (str) 'a' arcs-focused, 'g' graph-focused, 'n' node-focused. See GraphObject.merge for details.
        :param aggregation_mode: (str) incoming message aggregation mode: 'sum', 'average', 'normalized'. See GraphObject.merge for details.
        :param batch_size: (int) batch size for merging graphs data.
        :param shuffle: (bool) if True, at the end of the epoch, data is shuffled. No shuffling is performed otherwise. """
        self.data = graphs if isinstance(graphs, list) else [graphs]
        self.focus = focus
        self.aggregation_mode = aggregation_mode
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.dtype = tf.keras.backend.floatx()
        self.build_batches()

    # -----------------------------------------------------------------------------------------------------------------
    def build_batches(self):
        """ Create batches from sequencer data. """
        graphs = [self.merge(self.data[i * self.batch_size: (i + 1) * self.batch_size], focus=self.focus,
                             aggregation_mode=self.aggregation_mode) for i in range(len(self))]
        self.graph_tensors = [self.to_graph_tensor(g) for g in graphs]

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self):
        """ COPY METHOD

        :return: a Deep Copy of the GraphSequencer instance. """
        config = self.get_config()
        config["graphs"] = [g.copy() for g in config["graphs"]]
        return self.from_config(config)

    ## CONFIG METHODs #################################################################################################
    def get_config(self):
        """ Get configuration dictionary. To be used with from_config().
        It is good practice providing this method to user. """
        return {"graphs": self.data,
                "focus": self.focus,
                "aggregation_mode": self.aggregation_mode,
                "batch_size": self.batch_size,
                "shuffle": self.shuffle}

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config, **kwargs):
        """ Create class from configuration dictionary. To be used with get_config().
        It is good practice providing this method to user. """
        return cls(**config)

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self):
        """ Representation string for the instance of GraphSequencer. """
        problem = {'a': 'edge', 'n': 'node', 'g': 'graph'}[self.focus]
        return f"graph_sequencer(type=multiple {problem}-focused, len={len(self)}, " \
               f"aggregation='{self.aggregation_mode}', batch_size={self.batch_size}, shuffle={self.shuffle})"

    # -----------------------------------------------------------------------------------------------------------------
    def __str__(self):
        """ Representation string for the instance of GraphSequencer, for print() purpose. """
        return self.__repr__()

    ## SETTER and GETTER METHODs ######################################################################################
    def set_batch_size(self, new_batch_size):
        """ Modify batch size, then re-create batches. """
        self.batch_size = new_batch_size
        self.build_batches()

    # -----------------------------------------------------------------------------------------------------------------
    def get_batch(self, index):
        """ Return the single graph_tensor corresponding to the considered batch and its mask. """
        g = self.graph_tensors[index]
        return g, g.set_mask

    ## IMPLEMENTED ABSTRACT METHODs ###################################################################################
    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return int(np.ceil(len(self.data) / self.batch_size))

    # -----------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index):
        """ Get single batch data. """
        g, set_mask = self.get_batch(index)

        newaxis = lambda x: x[..., tf.newaxis]
        out = [g.nodes, g.arcs] + [newaxis(i) for i in [g.DIM_NODE_LABEL, g.set_mask, g.output_mask]] + \
              [(i.indices, newaxis(i.values), tf.constant(i.shape, dtype=tf.int64)) for i in [g.Adjacency, g.ArcNode, g.NodeGraph]]

        if self.focus == 'g': mask = tf.ones((g.targets.shape[0]), dtype=bool)
        else: mask = tf.boolean_mask(set_mask, g.output_mask)

        targets = tf.boolean_mask(g.targets, mask)
        sample_weight = tf.boolean_mask(g.sample_weight, mask)

        # out order:
        # nodes, arcs, dim_node_label, set_mask, output_mask, Adjacency, ArcNode, NodeGraph
        return out, targets, sample_weight

    # -----------------------------------------------------------------------------------------------------------------
    def on_epoch_end(self):
        """ Update data after each epoch. Rebuild batches if data is shuffled. """
        if self.shuffle:
            np.random.shuffle(self.data)
            self.build_batches()


#######################################################################################################################
### CLASS GRAPH SEQUENCER FOR SINGLE HOMOGENEOUS DATA ### FOR FEEDING THE MODEL DURING LEARNING PROCESS ###############
#######################################################################################################################
class SingleGraphSequencer(MultiGraphSequencer):
    """ GraphSequencer for dataset composed of only one single Homogeneous Graph. """

    # Specific function utilities.
    to_graph_tensor = classmethod(GraphTensor.fromGraphObject)

    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self,
                 graph: GraphObject,
                 focus: str,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """ CONSTRUCTOR

        :param graph: a single GraphObject element to be sequenced.
        :param focus: (str) 'a' arcs-focused, 'g' graph-focused, 'n' node-focused. See GraphObject.__init__ for details.
        :param batch_size: (int) batch size for set_mask_idx values.
        :param shuffle: (bool) if True, at the end of the epoch, set_mask_idx is shuffled. No shuffling is performed otherwise. """
        self.data = graph
        self.graph_tensor = self.to_graph_tensor(graph)
        self.focus = focus
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dtype = tf.keras.backend.floatx()

        self.set_mask_idx = np.argwhere(self.data.set_mask).squeeze()
        self.build_batches()

    # -----------------------------------------------------------------------------------------------------------------
    def build_batches(self):
        """ Create batches from sequencer data. """
        self.batch_masks = np.zeros((len(self), len(self.data.set_mask)), dtype=bool)
        for i in range(len(self)):
            self.batch_masks[i, self.set_mask_idx[i * self.batch_size: (i + 1) * self.batch_size]] = True

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self):
        """ COPY METHOD

        :return: a Deep Copy of the GraphSequencer instance. """
        config = self.get_config()
        config["graph"] = config["graph"].copy()
        return self.from_config(config)

    ## CONFIG METHODs #################################################################################################
    def get_config(self):
        """ Get configuration dictionary. To be used with from_config().
        It is good practice providing this method to user. """
        return {"graph": self.data,
                "focus": self.focus,
                "batch_size": self.batch_size,
                "shuffle": self.shuffle}

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self):
        """ Representation string for the instance of GraphSequencer. """
        problem = {'a': 'edge', 'n': 'node', 'g': 'graph'}[self.focus]
        return f"graph_sequencer(type=single {problem}-focused, " \
               f"len={len(self)}, batch_size={self.batch_size}, shuffle={self.shuffle})"

    ## SETTER and GETTER METHODs ######################################################################################
    def get_batch(self, index):
        """ Return the single graph_tensor and a mask for the considered batch. """
        return self.graph_tensor, tf.constant(self.batch_masks[index], dtype=bool)

    ## IMPLEMENTED ABSTRACT METHODs ###################################################################################
    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return int(np.ceil(np.sum(self.data.set_mask) / self.batch_size))

    # -----------------------------------------------------------------------------------------------------------------
    def on_epoch_end(self):
        """ Update set_mask indices after each epoch. Rebuild batches if set_mask indices are shuffled. """
        if self.shuffle:
            np.random.shuffle(self.set_mask_idx)
            self.build_batches()


#######################################################################################################################
### CLASS GRAPH SEQUENCER FOR MULTIPLE HETEROGENEOUS DATA ### FOR FEEDING THE MODEL DURING LEARNING PROCESS ###########
#######################################################################################################################
class CompositeMultiGraphSequencer(MultiGraphSequencer):
    """ GraphSequencer for dataset composed of multiple Heterogeneous Graphs. """

    # Specific function utilities.
    merge = classmethod(CompositeGraphObject.merge)
    to_graph_tensor = classmethod(CompositeGraphTensor.fromGraphObject)

    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self, graphs: list[CompositeGraphObject], *args, **kwargs):
        """ COSNTRUCTOR - re-defined only to hint graphs. """
        super().__init__(graphs, *args, **kwargs)

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self):
        """ Representation string for the instance of CompositeGraphSequencer. """
        return f"composite_{super().__repr__()}"

    ## IMPLEMENTED ABSTRACT METHODs ###################################################################################
    def __getitem__(self, index):
        """ Get single batch data. """

        out, target, sample_weight = super().__getitem__(index)

        g, set_mask = self.get_batch(index)

        newaxis = lambda x: x[..., tf.newaxis]
        out.insert(3,  newaxis(g.type_mask))
        out.insert(-3, [(ca.indices, newaxis(ca.values), tf.constant(ca.shape, dtype=tf.int64)) for ca in g.CompositeAdjacencies])

        # out order:
        # nodes, arcs, dim_node_label, type_mask, set_mask, output_mask, CompositeAdjacency, Adjacency, ArcNode, NodeGraph.
        return out, target, sample_weight



#######################################################################################################################
### CLASS GRAPH SEQUENCER FOR SINGLE HETEROGENEOUS DATA ### FOR FEEDING THE MODEL DURING LEARNING PROCESS #############
#######################################################################################################################
class CompositeSingleGraphSequencer(SingleGraphSequencer, CompositeMultiGraphSequencer):
    """ GraphSequencer for dataset composed of only  one single Heterogeneous Graph. """

    # Specific function utilities.
    to_graph_tensor = classmethod(CompositeGraphTensor.fromGraphObject)

    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self, graph: CompositeGraphObject, *args, **kwargs):
        """ Initialization - re-defined only to hint graph. """
        SingleGraphSequencer.__init__(self, graph, *args, **kwargs)

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self):
        """ Representation string for the instance of CompositeGraphSequencer. """
        return f"composite_{super().__repr__()}"
