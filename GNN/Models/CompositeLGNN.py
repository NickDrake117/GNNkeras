# codinf=utf-8
import tensorflow as tf

from GNN.Models.LGNN import LGNN
from GNN.Models.CompositeGNN import CompositeGNNnodeBased as GNNnodeBased, CompositeGNNgraphBased as GNNgraphBased


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

    @property
    def processing_function(self):
        # get processing function to retrieve state and output for the nodes of the graphs processed by the gnn layer.
        # Fundamental for graph-focused problem, since the output is referred to the entire graph, rather than to the graph nodes.
        # Since in CONSTRUCTOR GNNs type must be only one, type(self.gnns[0]) is exploited
        # and processing function is the same overall GNNs layers.
        return GNNnodeBased.Loop if self.GNN_CLASS._name == "graph" else self.GNN_CLASS.Loop

    ## LOOP METHODs ###################################################################################################
    def Loop(self, nodes, arcs, dim_node_features, type_mask, set_mask, output_mask, composite_adjacencies, adjacency, arcnode, nodegraph,
             training: bool = False) -> tuple[list[tf.Tensor], tf.Tensor, list[tf.Tensor]]:
        """ Process a single GraphTensor element, returning 3 lists of iterations, states and outputs. """
        constant_inputs = [type_mask, set_mask, output_mask, composite_adjacencies, adjacency, arcnode, nodegraph]

        # deep copy of nodes and arcs at time t==0.
        dtype = tf.keras.backend.floatx()
        nodes_0, arcs_0, d_0 = tf.constant(nodes, dtype=dtype), tf.constant(arcs, dtype=dtype), tf.constant(dim_node_features)

        # forward pass.
        K, states, outs = list(), list(), list()
        for idx, gnn in enumerate(self.gnns[:-1]):
            # process graph.
            k, state, out = self.processing_function(gnn, nodes, arcs, dim_node_features, *constant_inputs, training=training)

            # append new k, new states and new gnn output.
            K.append(k)
            states.append(state)
            outs.append(tf.sparse.sparse_dense_matmul(nodegraph, out, adjoint_a=True) if isinstance(gnn, GNNgraphBased) else out)

            # update graph with nodes' state and  nodes/arcs' output of the current GNN layer, to feed next GNN layer.
            nodes, arcs, dim_node_features = self.update_graph(nodes_0, arcs_0, d_0, set_mask, output_mask, state, out)

        # final GNN k, state and out values.
        k, state, out = self.gnns[-1].Loop(nodes, arcs, dim_node_features, *constant_inputs, training=training)

        # return 3 lists of Ks, states and gnn outputs, s.t. len == self.LAYERS.
        return K + [k], states + [state], outs + [out]
