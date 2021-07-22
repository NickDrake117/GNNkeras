from __future__ import annotations

from typing import Union

import numpy as np
import tensorflow as tf

from CompositeGNN.composite_graph_class import CompositeGraphObject, CompositeGraphTensor
from GNN.graph_class import GraphObject, GraphTensor


class GraphDataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 graphs: Union[GraphObject, list[GraphObject]],
                 problem_based: str,
                 aggregation_mode: str,
                 batch_size: int = 32,
                 shuffle: bool = True):
        '''Initialization'''
        self.data = graphs if isinstance(graphs, list) else [graphs]
        self.problem_based = problem_based
        self.aggregation_mode = aggregation_mode
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.length = int(np.ceil(len(self.data) / self.batch_size))
        self.dtype = tf.keras.backend.floatx()
        self.on_epoch_end()

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return self.length

    def __getitem__(self, index):
        '''Generate one batch of data'''
        g = self.graph_tensors[index]

        out = [g.nodes, g.arcs, g.set_mask[:, tf.newaxis], g.output_mask[:, tf.newaxis],
               (g.Adjacency.indices, g.Adjacency.values), (g.ArcNode.indices, g.ArcNode.values),
               g.NodeGraph]

        mask = tf.ones((g.targets.shape[0]), dtype=bool) if self.problem_based == 'g' else tf.boolean_mask(g.set_mask, g.output_mask)
        targets = tf.boolean_mask(g.targets, mask)
        sample_weights = tf.boolean_mask(g.sample_weights, mask)

        return out, targets, sample_weights

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        if self.shuffle: np.random.shuffle(self.data)
        graphs = [GraphObject.merge(self.data[i * self.batch_size: (i + 1) * self.batch_size], problem_based=self.problem_based,
                                    aggregation_mode=self.aggregation_mode) for i in range(self.length)]
        self.graph_tensors = [GraphTensor.fromGraphObject(g) for g in graphs]


################################################################################################
class CompositeGraphDataGenerator(GraphDataGenerator):

    def __getitem__(self, index):
        '''Generate one batch of data'''
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

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        if self.shuffle: np.random.shuffle(self.data)
        graphs = [CompositeGraphObject.merge(self.data[i * self.batch_size: (i + 1) * self.batch_size], problem_based=self.problem_based,
                                             aggregation_mode=self.aggregation_mode) for i in range(self.length)]
        self.graph_tensors = [CompositeGraphTensor.fromGraphObject(g) for g in graphs]
