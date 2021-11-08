from typing import Union

import numpy as np
import tensorflow as tf

from GNN.GraphGenerators import CompositeMultiGraphGenerator, CompositeSingleGraphGenerator
from GNN.composite_graph_class import CompositeGraphObject
from GNN.graph_class import GraphObject


#######################################################################################################################
### CLASS GRAPH GENERATORS FOR MULTIPLE HETEROGENEOUS DATA ### FOR FEEDING THE MODEL DURING LEARNING PROCESS ##########
#######################################################################################################################
class TransductiveMultiGraphGenerator(CompositeMultiGraphGenerator):
    """ prendo grafi omogenei multipli e li trasforma in grafi eterogenei multipli con tipi [non_transduttivi, trasduttivi] """

    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 graphs: Union[list[GraphObject], list[CompositeGraphObject]],
                 problem_based: str,
                 aggregation_mode: str,
                 transductive_rate: float = 0.5,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """ Initialization """
        self.transductive_rate = transductive_rate
        super().__init__(graphs, problem_based, aggregation_mode, batch_size, shuffle)

    # -----------------------------------------------------------------------------------------------------------------
    def get_transduction(self, g: GraphObject):
        """ get the transductive version of :param g: -> an heterogeneous graph with non-transductive/transductive nodes """
        transductive_node_mask = np.logical_and(g.set_mask, g.output_mask)

        indices = np.argwhere(transductive_node_mask).squeeze()
        np.random.shuffle(indices)

        non_transductive_number = int(np.ceil(np.sum(transductive_node_mask) * (1 - self.transductive_rate)))
        transductive_node_mask[indices[:non_transductive_number]] = False

        transductive_target_mask = transductive_node_mask[g.output_mask]

        # new nodes/arcs label
        length = {'a': g.arcs.shape[0]}.get(self.problem_based, g.nodes.shape[0])
        labelplus = np.zeros((length, g.DIM_TARGET), dtype=self.dtype)
        labelplus[transductive_node_mask] = g.targets[transductive_target_mask]

        # nuove quantità per target, nodi, output, dim_nodes, type_mask ecc.
        nodes_new = np.concatenate([g.nodes, labelplus], axis=1)
        target_new = g.targets[np.logical_not(transductive_target_mask)]

        dim_node_label_new = (g.DIM_NODE_LABEL, g.DIM_NODE_LABEL + g.DIM_TARGET)

        type_mask = np.zeros((g.nodes.shape[0], 2), dtype=bool)
        type_mask[transductive_node_mask, 1] = True
        type_mask[:, 0] = np.logical_not(type_mask[:, 1])

        output_mask_new = g.output_mask.copy()
        output_mask_new[transductive_node_mask] = False

        return CompositeGraphObject(arcs=g.getArcs(), nodes=nodes_new, targets=target_new, type_mask=type_mask,
                                    dim_node_labels=dim_node_label_new, problem_based=self.problem_based,
                                    set_mask=g.getSetMask(), output_mask=output_mask_new)

    def __repr__(self):
        problem = {'a': 'edge', 'n': 'node', 'g': 'graph'}[self.problem_based]
        return f"transductive_graph_generator(multiple {problem}-based, len={len(self)}, " \
               f"transductive_rate={self.transductive_rate}, aggregation='{self.aggregation_mode}', " \
               f"batch_size={self.batch_size}, shuffle={self.shuffle})"

    # -----------------------------------------------------------------------------------------------------------------
    def on_epoch_end(self):
        """ Updates graphs after each epoch: get the transductive (eterogeneous) version of graphs and then merge """
        if self.shuffle: np.random.shuffle(self.data)
        graphs = [self.get_transduction(g) for g in self.data]
        graphs = [self.merge(graphs[i * self.batch_size: (i + 1) * self.batch_size], problem_based=self.problem_based,
                             aggregation_mode=self.aggregation_mode) for i in range(len(self))]
        self.graph_tensors = [self.to_graph_tensor(g) for g in graphs]


#######################################################################################################################
### CLASS GRAPH GENERATORS FOR SINGLE HETEROGENEOUS DATA ### FOR FEEDING THE MODEL DURING LEARNING PROCESS ############
#######################################################################################################################
class TransductiveSingleGraphGenerator(TransductiveMultiGraphGenerator, CompositeSingleGraphGenerator):
    def __init__(self,
                 graph: GraphObject,
                 problem_based: str,
                 transductive_rate: float = 0.5,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """ Initialization """
        self.data = graph
        self.problem_based = problem_based
        self.transductive_rate = transductive_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dtype = tf.keras.backend.floatx()

        self.gen_set_mask_idx = np.argwhere(self.data.set_mask).squeeze()
        self.on_epoch_end()

    # -----------------------------------------------------------------------------------------------------------------
    def __repr__(self):
        problem = {'a': 'edge', 'n': 'node', 'g': 'graph'}[self.problem_based]
        return f"transductive_graph_generator(type=single {problem}-based, " \
               f"len={len(self)}, transductive_rate={self.transductive_rate}, " \
               f"batch_size={self.batch_size}, shuffle={self.shuffle})"

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self):
        new_gen = self.__class__(self.data.copy(), self.problem_based, self.trasductive_rate, self.batch_size, False)
        new_gen.shuffle = self.shuffle
        return new_gen

    # -----------------------------------------------------------------------------------------------------------------
    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        CompositeSingleGraphGenerator.on_epoch_end(self)

        # overwrite graph_tensor, since at the end of the epoch the transductive nodes set is changed
        g = self.get_transduction(self.data)
        self.graph_tensor = self.to_graph_tensor(g)
