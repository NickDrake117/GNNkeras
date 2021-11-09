import numpy as np
import tensorflow as tf

from GNN.composite_graph_class import CompositeGraphObject, CompositeGraphTensor
from GNN.graph_class import GraphObject, GraphTensor


#######################################################################################################################
### CLASS GRAPH GENERATORS FOR MULTIPLE HOMOGENEOUS DATA ### FOR FEEDING THE MODEL DURING LEARNING PROCESS ### BASE ###
#######################################################################################################################
class MultiGraphGenerator(tf.keras.utils.Sequence):
    """ Generator for dataset composed of multiple Homogeneous Graphs """

    # specific function utilities
    merge = GraphObject.merge
    to_graph_tensor = GraphTensor.fromGraphObject

    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 graphs: list[GraphObject],
                 problem_based: str,
                 aggregation_mode: str,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """ Initialization """
        self.data = graphs if isinstance(graphs, list) else [graphs]
        self.problem_based = problem_based
        self.aggregation_mode = aggregation_mode
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dtype = tf.keras.backend.floatx()
        self.build_batches()

    # -----------------------------------------------------------------------------------------------------------------
    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.ceil(len(self.data) / self.batch_size))

    # -----------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index):
        """ Generate one batch of data """
        g, set_mask = self.get_batch(index)

        out = [g.nodes, g.arcs, set_mask[:, tf.newaxis], g.output_mask[:, tf.newaxis],
               (g.Adjacency.indices, g.Adjacency.values), (g.ArcNode.indices, g.ArcNode.values),
               g.NodeGraph]

        if self.problem_based == 'g': mask = tf.ones((g.targets.shape[0]), dtype=bool)
        else: mask = tf.boolean_mask(set_mask, g.output_mask)

        targets = tf.boolean_mask(g.targets, mask)
        sample_weights = tf.boolean_mask(g.sample_weights, mask)

        return out, targets, sample_weights

    # -----------------------------------------------------------------------------------------------------------------
    def __repr__(self):
        problem = {'a': 'edge', 'n': 'node', 'g': 'graph'}[self.problem_based]
        return f"graph_generator(type=multiple {problem}-based', len={len(self)}, " \
               f"aggregation='{self.aggregation_mode}', batch_size={self.batch_size}, shuffle={self.shuffle})"

    # -----------------------------------------------------------------------------------------------------------------
    def __str__(self):
        return self.__repr__()

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self):
        """ copy method """
        new_gen = self.__class__([i.copy() for i in self.data], self.problem_based, self.aggregation_mode, self.batch_size, False)
        new_gen.shuffle = self.shuffle
        return new_gen

    # -----------------------------------------------------------------------------------------------------------------
    def set_batch_size(self, new_batch_size):
        """ modify batch size, then re-create batches """
        self.batch_size = new_batch_size
        self.on_epoch_end()

    # -----------------------------------------------------------------------------------------------------------------
    def get_batch(self, index):
        """ return the single graph_tensor corresponding to the considered batch and its mask """
        g = self.graph_tensors[index]
        return g, g.set_mask

    # -----------------------------------------------------------------------------------------------------------------
    def build_batches(self):
        graphs = [self.merge(self.data[i * self.batch_size: (i + 1) * self.batch_size], problem_based=self.problem_based,
                             aggregation_mode=self.aggregation_mode) for i in range(len(self))]
        self.graph_tensors = [self.to_graph_tensor(g) for g in graphs]

    # -----------------------------------------------------------------------------------------------------------------
    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        if self.shuffle:
            np.random.shuffle(self.data)
            self.build_batches()


#######################################################################################################################
### CLASS GRAPH GENERATORS FOR SINGLE HOMOGENEOUS DATA ### FOR FEEDING THE MODEL DURING LEARNING PROCESS ##############
#######################################################################################################################
class SingleGraphGenerator(MultiGraphGenerator):  # (tf.keras.utils.Sequence, GraphGenerator):
    """ Generator for dataset composed of only one single Homogeneous Graph """

    # specific function utilities
    to_graph_tensor = GraphTensor.fromGraphObject

    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 graph: GraphObject,
                 problem_based: str,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """ Initialization """
        self.data = graph
        self.graph_tensor = self.to_graph_tensor(graph)
        self.problem_based = problem_based
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dtype = tf.keras.backend.floatx()

        self.set_mask_idx = np.argwhere(self.data.set_mask).squeeze()
        self.build_batches()

    # -----------------------------------------------------------------------------------------------------------------
    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.ceil(np.sum(self.data.set_mask) / self.batch_size))

    # -----------------------------------------------------------------------------------------------------------------
    def __repr__(self):
        problem = {'a': 'edge', 'n': 'node', 'g': 'graph'}[self.problem_based]
        return f"graph_generator(type=single {problem}-based, " \
               f"len={len(self)}, batch_size={self.batch_size}, shuffle={self.shuffle})"

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self):
        """ copy method """
        new_gen = self.__class__(self.data.copy(), self.problem_based, self.batch_size, False)
        new_gen.shuffle = self.shuffle
        return new_gen

    # -----------------------------------------------------------------------------------------------------------------
    def get_batch(self, index):
        """ return the single graph_tensor and a mask for the considered batch """
        return self.graph_tensor, tf.constant(self.batch_masks[index], dtype=bool)

    # -----------------------------------------------------------------------------------------------------------------
    def build_batches(self):
        self.batch_masks = np.zeros((len(self), len(self.data.set_mask)), dtype=bool)
        for i in range(len(self)): self.batch_masks[i, self.set_mask_idx[i * self.batch_size: (i + 1) * self.batch_size]] = True

    # -----------------------------------------------------------------------------------------------------------------
    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        if self.shuffle:
            np.random.shuffle(self.set_mask_idx)
            self.build_batches()


#######################################################################################################################
### CLASS GRAPH GENERATORS FOR MULTIPLE HETEROGENEOUS DATA ### FOR FEEDING THE MODEL DURING LEARNING PROCESS ##########
#######################################################################################################################
class CompositeMultiGraphGenerator(MultiGraphGenerator):
    """ Generator for dataset composed of multiple Heterogeneous Graphs """

    # specific function utilities
    merge = CompositeGraphObject.merge
    to_graph_tensor = CompositeGraphTensor.fromGraphObject

    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self, graphs: list[CompositeGraphObject], *args):
        """ Initialization - re-defined only to hint graphs """
        super().__init__(graphs, *args)

    # -----------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index):
        """ Generate one batch of data """
        g, set_mask = self.get_batch(index)

        out = [g.nodes, g.arcs, g.DIM_NODE_LABEL[:, tf.newaxis], g.type_mask,
               g.set_mask[:, tf.newaxis], g.output_mask[:, tf.newaxis],
               (g.Adjacency.indices, g.Adjacency.values),
               [(ca.indices, ca.values[:, tf.newaxis]) for ca in g.CompositeAdjacencies],
               (g.ArcNode.indices, g.ArcNode.values),
               g.NodeGraph]

        if self.problem_based == 'g': mask = tf.ones((g.targets.shape[0]), dtype=bool)
        else: mask = tf.boolean_mask(set_mask, g.output_mask)

        targets = tf.boolean_mask(g.targets, mask)
        sample_weights = tf.boolean_mask(g.sample_weights, mask)

        return out, targets, sample_weights

    # -----------------------------------------------------------------------------------------------------------------
    def __repr__(self):
        problem = {'a': 'edge', 'n': 'node', 'g': 'graph'}[self.problem_based]
        return f"composite_graph_generator(type=multiple {problem}-based, len={len(self)}, " \
               f"aggregation={self.aggregation_mode}, batch_size={self.batch_size}, shuffle={self.shuffle})"


#######################################################################################################################
### CLASS GRAPH GENERATORS FOR SINGLE HETEROGENEOUS DATA ### FOR FEEDING THE MODEL DURING LEARNING PROCESS ############
#######################################################################################################################
class CompositeSingleGraphGenerator(SingleGraphGenerator, CompositeMultiGraphGenerator):
    """ Generator for dataset composed of only  one single Heterogeneous Graph """

    # specific function utilities
    to_graph_tensor = CompositeGraphTensor.fromGraphObject

    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self, graph: CompositeGraphObject, *args):
        """ Initialization - re-defined only to hint graph """
        SingleGraphGenerator.__init__(self, graph, *args)

    # -----------------------------------------------------------------------------------------------------------------
    def __repr__(self):
        problem = {'a': 'edge', 'n': 'node', 'g': 'graph'}[self.problem_based]
        return f"composite_graph_generator(type=single {problem}-based, " \
               f"len={len(self)}, batch_size={self.batch_size}, shuffle={self.shuffle})"
