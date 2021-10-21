from __future__ import annotations

from typing import Union

import numpy as np
import tensorflow as tf

from GNN.composite_graph_class import CompositeGraphObject, CompositeGraphTensor
from GNN.graph_class import GraphObject, GraphTensor


#######################################################################################################################
### CLASS GRAPH GENERATORS ### FOR FEEDING THE MODEL DURING LEARNING PROCESS ##########################################
#######################################################################################################################
class GraphDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 graphs: Union[GraphObject, list[GraphObject]],
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
        self.on_epoch_end()

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self):
        new_gen = self.__class__([i.copy() for i in self.data], self.problem_based, self.aggregation_mode, self.batch_size, False)
        new_gen.shuffle = self.shuffle
        return new_gen

    # -----------------------------------------------------------------------------------------------------------------
    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.ceil(len(self.data) / self.batch_size))

    # -----------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index):
        """ Generate one batch of data """
        g = self.graph_tensors[index]

        out = [g.nodes, g.arcs, g.set_mask[:, tf.newaxis], g.output_mask[:, tf.newaxis],
               (g.Adjacency.indices, g.Adjacency.values), (g.ArcNode.indices, g.ArcNode.values),
               g.NodeGraph]

        mask = tf.ones((g.targets.shape[0]), dtype=bool) if self.problem_based == 'g' else tf.boolean_mask(g.set_mask, g.output_mask)
        targets = tf.boolean_mask(g.targets, mask)
        sample_weights = tf.boolean_mask(g.sample_weights, mask)

        return out, targets, sample_weights

    # -----------------------------------------------------------------------------------------------------------------
    def set_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size
        self.on_epoch_end()

    # -----------------------------------------------------------------------------------------------------------------
    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        if self.shuffle: np.random.shuffle(self.data)
        graphs = [GraphObject.merge(self.data[i * self.batch_size: (i + 1) * self.batch_size], problem_based=self.problem_based,
                                    aggregation_mode=self.aggregation_mode) for i in range(len(self))]
        self.graph_tensors = [GraphTensor.fromGraphObject(g) for g in graphs]


#######################################################################################################################
class SingleGraphDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 graph: GraphObject,
                 problem_based: str,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """ Initialization """
        self.data = graph
        self.graph_tensor = GraphTensor.fromGraphObject(graph)
        self.problem_based = problem_based
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dtype = tf.keras.backend.floatx()

        self.gen_set_mask_idx = np.argwhere(self.data.set_mask).squeeze()
        self.on_epoch_end()

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self):
        new_gen = self.__class__(self.data.copy(), self.problem_based, self.batch_size, False)
        new_gen.shuffle = self.shuffle
        return new_gen

    # -----------------------------------------------------------------------------------------------------------------
    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.ceil(np.sum(self.data.set_mask) / self.batch_size))

    # -----------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index):
        """ Generate one batch of data """
        g = self.graph_tensor
        batch_set_mask = tf.constant(self.batch_masks[index, :], dtype=bool)

        out = [g.nodes, g.arcs, batch_set_mask[:, tf.newaxis], g.output_mask[:, tf.newaxis],
               (g.Adjacency.indices, g.Adjacency.values), (g.ArcNode.indices, g.ArcNode.values),
               g.NodeGraph]

        mask = tf.ones((g.targets.shape[0]), dtype=bool) if self.problem_based == 'g' else tf.boolean_mask(batch_set_mask, g.output_mask)
        targets = tf.boolean_mask(g.targets, mask)
        sample_weights = tf.boolean_mask(g.sample_weights, mask)

        return out, targets, sample_weights

    # -----------------------------------------------------------------------------------------------------------------
    def set_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size
        self.on_epoch_end()

    # -----------------------------------------------------------------------------------------------------------------
    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        set_mask = self.gen_set_mask_idx.copy()
        if self.shuffle: np.random.shuffle(set_mask)

        self.batch_masks = np.zeros((len(self), len(self.data.set_mask)), dtype=bool)
        for i in range(len(self)): self.batch_masks[i, set_mask[i * self.batch_size: (i + 1) * self.batch_size]] = True


#######################################################################################################################
### CLASS GRAPH TENSOR GENERATORS ### FOR FEEDING THE MODEL DURING LEARNING PROCESS ###################################
#######################################################################################################################
class CompositeGraphDataGenerator(GraphDataGenerator):
    def __getitem__(self, index):
        """ Generate one batch of data """
        g = self.graph_tensors[index]

        out = [g.nodes, g.arcs, g.DIM_NODE_LABELS[:, tf.newaxis], g.type_mask,
               g.set_mask[:, tf.newaxis], g.output_mask[:, tf.newaxis],
               (g.Adjacency.indices, g.Adjacency.values),
               [(ca.indices, ca.values[:, tf.newaxis]) for ca in g.CompositeAdjacencies],
               (g.ArcNode.indices, g.ArcNode.values),
               g.NodeGraph]

        mask = tf.ones((g.targets.shape[0]), dtype=bool) if self.problem_based == 'g' else tf.boolean_mask(g.set_mask, g.output_mask)
        targets = tf.boolean_mask(g.targets, mask)
        sample_weights = tf.boolean_mask(g.sample_weights, mask)

        return out, targets, sample_weights

    # -----------------------------------------------------------------------------------------------------------------
    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        if self.shuffle: np.random.shuffle(self.data)
        graphs = [CompositeGraphObject.merge(self.data[i * self.batch_size: (i + 1) * self.batch_size], problem_based=self.problem_based,
                                             aggregation_mode=self.aggregation_mode) for i in range(len(self))]
        self.graph_tensors = [CompositeGraphTensor.fromGraphObject(g) for g in graphs]