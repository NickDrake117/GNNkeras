# codinf=utf-8
import numpy as np
import tensorflow as tf

from GNN.Sequencers.GraphSequencers import CompositeMultiGraphSequencer, CompositeSingleGraphSequencer
from GNN.composite_graph_class import CompositeGraphObject
from GNN.graph_class import GraphObject


#######################################################################################################################
### CLASS GRAPH SEQUENCER FOR MULTIPLE HOMOGENEOUS DATA ### FOR FEEDING THE MODEL DURING LEARNING PROCESS #############
#######################################################################################################################
class TransductiveMultiGraphSequencer(CompositeMultiGraphSequencer):
    """ GraphSequencer for dataset composed of multiple Homogeneous Graphs,
    which are converted into Heterogeneous Graphs, with "transductive" and "non-transductive" node types. """

    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self,
                 graphs: list[GraphObject],
                 focus: str,
                 aggregation_mode: str,
                 transductive_rate: float = 0.5,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """ CONSTRUCTOR

        :param graphs: a list of GraphObject elements to be sequenced.
        :param focus: (str) 'a' arcs-focused, 'g' graph-focused, 'n' node-focused. See GraphObject.merge for details.
        :param aggregation_mode: (str) incoming message aggregation mode: 'sum', 'average', 'normalized'. See GraphObject.merge for details.
        :param transductive_rate: (float) targeted nodes' rate to be considered as transductive nodes.
        :param batch_size: (int) batch size for merging graphs data.
        :param shuffle: (bool) if True, at the end of the epoch, data is shuffled. No shuffling is performed otherwise. """
        self.graph_objects = graphs
        self.transductive_rate = transductive_rate

        gs = [self.get_transduction(g, transductive_rate, focus, tf.keras.backend.floatx()) for g in graphs]
        super().__init__(gs, focus, aggregation_mode,batch_size, shuffle)

    ## CONFIG METHODs #################################################################################################
    def get_config(self):
        """ Get configuration dictionary. To be used with from_config().
        It is good practice providing this method to user. """
        config = super().get_config()
        config["transductive_rate"] = self.transductive_rate
        return config

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self):
        """ Representation string for the instance of GraphSequencer. """
        problem = {'a': 'edge', 'n': 'node', 'g': 'graph'}[self.focus]
        return f"transductive_graph_sequencer(multiple {problem}-focused, len={len(self)}, " \
               f"transductive_rate={self.transductive_rate}, aggregation='{self.aggregation_mode}', " \
               f"batch_size={self.batch_size}, shuffle={self.shuffle})"

    ## IMPLEMENTED ABSTRACT METHODs ###################################################################################
    def on_epoch_end(self):
        """ Update transductive data after each epoch. Rebuild batches if data is shuffled. """
        self.data = [self.get_transduction(g, self.transductive_rate, self.focus, self.dtype) for g in self.graph_objects]
        super().on_epoch_end()

    ## STATIC METHODS #################################################################################################
    @staticmethod
    def get_transduction(g: GraphObject, transductive_rate: float, focus: str, dtype):
        """ get the transductive version of :param g: -> an heterogeneous graph with non-transductive/transductive nodes. """
        transductive_node_mask = np.logical_and(g.set_mask, g.output_mask)

        indices = np.argwhere(transductive_node_mask).squeeze()
        np.random.shuffle(indices)

        non_transductive_number = int(np.ceil(np.sum(transductive_node_mask) * (1 - transductive_rate)))
        transductive_node_mask[indices[:non_transductive_number]] = False

        transductive_target_mask = transductive_node_mask[g.output_mask]

        # new nodes/arcs label.
        length = g.arcs.shape[0] if focus == 'a' else g.nodes.shape[0]
        labelplus = np.zeros((length, g.DIM_TARGET), dtype=dtype)
        labelplus[transductive_node_mask] = g.targets[transductive_target_mask]

        # new version of target, nodes, output_mask, dim_node_label, type_mask etc.
        nodes_new = np.concatenate([g.nodes, labelplus], axis=1)
        target_new = g.targets[np.logical_not(transductive_target_mask)]

        dim_node_label_new = (g.DIM_NODE_LABEL, g.DIM_NODE_LABEL + g.DIM_TARGET)

        type_mask = np.zeros((g.nodes.shape[0], 2), dtype=bool)
        type_mask[transductive_node_mask, 1] = True
        type_mask[:, 0] = np.logical_not(type_mask[:, 1])

        output_mask_new = g.output_mask.copy()
        output_mask_new[transductive_node_mask] = False

        return CompositeGraphObject(arcs=g.getArcs(), nodes=nodes_new, targets=target_new, type_mask=type_mask,
                                    dim_node_label=dim_node_label_new, focus=focus,
                                    set_mask=g.getSetMask(), output_mask=output_mask_new)

#######################################################################################################################
### CLASS GRAPH SEQUENCER FOR SINGLE HOMOGENEOUS DATA ### FOR FEEDING THE MODEL DURING LEARNING PROCESS ###############
#######################################################################################################################
class TransductiveSingleGraphSequencer(TransductiveMultiGraphSequencer, CompositeSingleGraphSequencer):
    """ Sequencer for dataset composed of only one single Homogeneous Graph
    The Homogeneous Graph is translated into an Heterogeneous Graph, with nodes' type [transductive, non_transductive]. """

    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self,
                 graph: GraphObject,
                 focus: str,
                 transductive_rate: float = 0.5,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """ CONSTRUCTOR
        :param graph: a single GraphObject element to be sequenced.
        :param focus: (str) 'a' arcs-focused, 'g' graph-focused, 'n' node-focused. See GraphObject.__init__ for details.
        :param transductive_rate: (float) targeted nodes' rate to be considered as transductive nodes.
        :param batch_size: (int) batch size for set_mask_idx values.
        :param shuffle: (bool) if True, at the end of the epoch, set_mask_idx is shuffled. No shuffling is performed otherwise. """
        self.graph_object = graph
        self.transductive_rate = transductive_rate

        g = self.get_transduction(graph, transductive_rate, focus, tf.keras.backend.floatx())
        CompositeSingleGraphSequencer.__init__(self, g, focus, batch_size, shuffle)

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self):
        """ COPY METHOD

        :return: a Deep Copy of the GraphSequencer instance. """
        new_gen = self.__class__(self.data.copy(), self.focus, self.trasductive_rate, self.batch_size, False)
        new_gen.shuffle = self.shuffle
        return new_gen

    ## CONFIG METHODs #################################################################################################
    def get_config(self):
        """ Get configuration dictionary. To be used with from_config().
        It is good practice providing this method to user. """
        config = super().get_config()
        config["transductive_rate"] = self.transductive_rate
        return config

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self):
        """ Representation string for the instance of GraphSequencer. """
        problem = {'a': 'edge', 'n': 'node', 'g': 'graph'}[self.focus]
        return f"transductive_graph_sequencer(type=single {problem}-focused, " \
               f"len={len(self)}, transductive_rate={self.transductive_rate}, " \
               f"batch_size={self.batch_size}, shuffle={self.shuffle})"

    ## IMPLEMENTED ABSTRACT METHODs ###################################################################################
    def on_epoch_end(self):
        """ Update transductive and set_mask indices after each epoch. Rebuild batches if set_mask indices are shuffled. """
        g = self.get_transduction(self.graph_object, self.transductive_rate, self.focus, self.dtype)
        self.graph_tensor = self.to_graph_tensor(g)
        CompositeSingleGraphSequencer.on_epoch_end(self)