from typing import Union

import numpy as np
import tensorflow as tf

from GNN.Sequencers.GraphSequencers import CompositeMultiGraphSequencer, CompositeSingleGraphSequencer
from GNN.composite_graph_class import CompositeGraphObject
from GNN.graph_class import GraphObject


#######################################################################################################################
### CLASS GRAPH GENERATORS FOR MULTIPLE HETEROGENEOUS DATA ### FOR FEEDING THE MODEL DURING LEARNING PROCESS ##########
#######################################################################################################################
class TransductiveMultiGraphSequencer(CompositeMultiGraphSequencer):
    """ Sequencer for dataset composed of multiple Homogeneous Graphs
    The Homogeneous Graphs are converted into Heterogeneous Graphs, with nodes' types [transductive, non_transductive] """

    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 graphs: list[GraphObject],
                 problem_based: str,
                 aggregation_mode: str,
                 transductive_rate: float = 0.5,
                 *args, **kwargs):
        """ Initialization """
        self.graph_objects = graphs
        self.transductive_rate = transductive_rate

        gs = [self.get_transduction(g, transductive_rate, problem_based, tf.keras.backend.floatx()) for g in graphs]
        super().__init__(gs, problem_based, aggregation_mode, *args, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_transduction(g: GraphObject, transductive_rate: float, problem_based: str, dtype):
        """ get the transductive version of :param g: -> an heterogeneous graph with non-transductive/transductive nodes """
        transductive_node_mask = np.logical_and(g.set_mask, g.output_mask)

        indices = np.argwhere(transductive_node_mask).squeeze()
        np.random.shuffle(indices)

        non_transductive_number = int(np.ceil(np.sum(transductive_node_mask) * (1 - transductive_rate)))
        transductive_node_mask[indices[:non_transductive_number]] = False

        transductive_target_mask = transductive_node_mask[g.output_mask]

        # new nodes/arcs label
        length = {'a': g.arcs.shape[0]}.get(problem_based, g.nodes.shape[0])
        labelplus = np.zeros((length, g.DIM_TARGET), dtype=dtype)
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
                                    dim_node_labels=dim_node_label_new, problem_based=problem_based,
                                    set_mask=g.getSetMask(), output_mask=output_mask_new)

    # -----------------------------------------------------------------------------------------------------------------
    def __repr__(self):
        problem = {'a': 'edge', 'n': 'node', 'g': 'graph'}[self.problem_based]
        return f"transductive_graph_generator(multiple {problem}-based, len={len(self)}, " \
               f"transductive_rate={self.transductive_rate}, aggregation='{self.aggregation_mode}', " \
               f"batch_size={self.batch_size}, shuffle={self.shuffle})"

    # -----------------------------------------------------------------------------------------------------------------
    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.data = [self.get_transduction(g, self.transductive_rate, self.problem_based, self.dtype) for g in self.graph_objects]
        super().on_epoch_end()


#######################################################################################################################
### CLASS GRAPH GENERATORS FOR SINGLE HETEROGENEOUS DATA ### FOR FEEDING THE MODEL DURING LEARNING PROCESS ############
#######################################################################################################################
class TransductiveSingleGraphSequencer(TransductiveMultiGraphSequencer, CompositeSingleGraphSequencer):
    """ Sequencer for dataset composed of only one single Homogeneous Graph
    The Homogeneous Graph is translated into an Heterogeneous Graph, with nodes' type [transductive, non_transductive] """

    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 graph: GraphObject,
                 problem_based: str,
                 transductive_rate: float = 0.5,
                 *args, **kwargs):
        """ Initialization """
        self.graph_object = graph
        self.transductive_rate = transductive_rate

        g = self.get_transduction(graph, transductive_rate, problem_based, tf.keras.backend.floatx())
        CompositeSingleGraphSequencer.__init__(self, g, problem_based, *args, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    def __repr__(self):
        problem = {'a': 'edge', 'n': 'node', 'g': 'graph'}[self.problem_based]
        return f"transductive_graph_generator(type=single {problem}-based, " \
               f"len={len(self)}, transductive_rate={self.transductive_rate}, " \
               f"batch_size={self.batch_size}, shuffle={self.shuffle})"

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self):
        """ copy method - return a deep copy of the generator """
        new_gen = self.__class__(self.data.copy(), self.problem_based, self.trasductive_rate, self.batch_size, False)
        new_gen.shuffle = self.shuffle
        return new_gen

    # -----------------------------------------------------------------------------------------------------------------
    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        g = self.get_transduction(self.graph_object, self.transductive_rate, self.problem_based, self.dtype)
        self.graph_tensor = self.to_graph_tensor(g)
        CompositeSingleGraphSequencer.on_epoch_end(self)