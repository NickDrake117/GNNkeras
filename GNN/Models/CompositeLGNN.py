# codinf=utf-8
import tensorflow as tf

from GNN.Models.LGNN import LGNN
from GNN.Models.CompositeGNN import CompositeGNNnodeBased as GNNnodeBased, \
    CompositeGNNarcBased as GNNarcBased, CompositeGNNgraphBased as GNNgraphBased


#######################################################################################################################
### CLASS LGNN - GENERAL ##############################################################################################
#######################################################################################################################
class CompositeLGNN(LGNN):
    """ Composite Layered Graph Neural Network (CLGNN) model for node-focused, arc-focused or graph-focused applications. """

    ## REPR METHODs ###################################################################################################
    def __repr__(self):
        return f"Composite{super().__repr__()}"

    ## STATIC METHODs #################################################################################################
    process_inputs = staticmethod(GNNnodeBased.process_inputs)
    __gnnClass__ = staticmethod(lambda x: {GNNnodeBased:"node", GNNarcBased: "arc", GNNgraphBased: "graph"}[x])
    __gnnClassLoader__ = staticmethod(lambda x: {"node": GNNnodeBased, "arc": GNNarcBased, "graph": GNNgraphBased}[x])

    ## LOOP METHODs ###################################################################################################
    def Loop(self, nodes, arcs, dim_node_label, type_mask, set_mask, output_mask, composite_adjacencies, adjacency, arcnode, nodegraph,
             training: bool = False) -> tuple[list[tf.Tensor], tf.Tensor, list[tf.Tensor]]:
        """ Process a single GraphTensor element, returning 3 lists of iterations, states and outputs. """
        constant_inputs = [type_mask, set_mask, output_mask, composite_adjacencies, adjacency, arcnode, nodegraph]

        # get processing function to retrieve state and output for the nodes of the graphs processed by the gnn layer.
        # Fundamental for graph-focused problem, since the output is referred to the entire graph, rather than to the graph nodes.
        # Since in CONSTRUCTOR GNNs type must be only one, type(self.gnns[0]) is exploited and processing function is the same overall GNNs layers.
        processing_function = self.__gnnClassLoader__("arc").Loop if self.gnns[0].name == "arc" else self.__gnnClassLoader__("node").Loop

        # deep copy of nodes and arcs at time t==0.
        dtype = tf.keras.backend.floatx()
        nodes_0, arcs_0 = tf.constant(nodes, dtype=dtype), tf.constant(arcs, dtype=dtype)

        # forward pass.
        K, states, outs = list(), list(), list()
        for idx, gnn in enumerate(self.gnns[:-1]):
            # process graph.
            k, state, out = processing_function(gnn, nodes, arcs, dim_node_label, *constant_inputs, training=training)

            # append new k, new states and new gnn output.
            K.append(k)
            states.append(state)
            outs.append(tf.sparse.sparse_dense_matmul(nodegraph, out, adjoint_a=True) if isinstance(gnn, GNNgraphBased) else out)

            # update graph with nodes' state and  nodes/arcs' output of the current GNN layer, to feed next GNN layer.
            nodes, arcs, dim_node_label = self.update_graph(nodes_0, arcs_0, dim_node_label, set_mask, output_mask, state, out)

        # final GNN k, state and out values.
        k, state, out = self.gnns[-1].Loop(nodes, arcs, dim_node_label, *constant_inputs, training=training)

        # return 3 lists of Ks, states and gnn outputs, s.t. len == self.LAYERS.
        return K + [k], states + [state], outs + [out]
