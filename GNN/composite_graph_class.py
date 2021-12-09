# coding=utf-8
import sys

import numpy as np
import tensorflow as tf

from scipy.sparse import coo_matrix
from GNN.graph_class import GraphObject, GraphTensor


#######################################################################################################################
## COMPOSITE GRAPH OBJECT CLASS #######################################################################################
#######################################################################################################################
class CompositeGraphObject(GraphObject):
    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self, nodes, arcs, targets, type_mask, dim_node_label, *args, **kwargs):
        """ CONSTRUCTOR METHOD

        :param nodes: Ordered Nodes Matrix where nodes[i] = [Node Label].
        :param arcs: Ordered Arcs Matrix where arcs[i] = [ID Node From | ID NodeTo | Arc Label].
        :param targets: Targets Array with shape (Num of targeted example [nodes or arcs], dim_target example).
        :param type_mask: boolean np.array with shape (Num of nodes, Num of node's types). i-th columns refers to dim_node_label[i]
        :param dim_node_label: (list/tuple) with len == Num of node's types. i-th element defines label dimension of nodes of type i.
        :param problem_based: (str) define the problem on which graph is used: 'a' arcs-based, 'g' graph-based, 'n' node-based.
        :param set_mask: Array of {0,1} to define arcs/nodes belonging to a set, when dataset == single GraphObject.
        :param output_mask: Array of {0,1} to define the sub-set of arcs/nodes whose target is known.
        :param sample_weights: target sample weight for loss computation. It can be int, float or numpy.array of ints or floats
            > If int, all targets are weighted as sample_weights * ones.
            > If numpy.array, len(sample_weights) and targets.shape[0] must agree.
        :param NodeGraph: Matrix (nodes.shape[0],{Num graphs or 1}) used only when problem_based=='g'.
        :param ArcNode: Matrix of shape (num_of_arcs, num_of_nodes) s.t. A[i,j]=value if arc[i,2]==node[j].
        :param aggregation_mode: (str) It defines the aggregation mode for the incoming message of a node using ArcNode and Adjacency:
            > 'average': elem(matrix)={0-1} -> matmul(m,A) gives the average of incoming messages, s.t. sum(A[:,i])=1;
            > 'normalized': elem(matrix)={0-1} -> matmul(m,A) gives the normalized message wrt the total number of g.nodes;
            > 'sum': elem(matrix)={0,1} -> matmul(m,A) gives the total sum of incoming messages. In this case Adjacency
            > 'composite_average': elem(matrix)={0-1} -> matmul(m,A) gives the average of incoming messages wrt nodes' type, s.t. sum(A[:,i])>=1;
        """
        super().__init__(nodes, arcs, targets, *args, **kwargs)

        # store dimensions: first two columns of arcs contain nodes indices
        self.DIM_NODE_LABEL = np.array(dim_node_label, ndmin=1, dtype=int)

        # type_mask[:,i] refers to nodes with DIM_NODE_LABEL[i] label dimension. Be careful when initializing a new graph
        self.type_mask = type_mask.astype(bool)

        # build Composite Adjacency Matrices. It is a list of Adjacency Matrix as long as the number of nodes' types.
        # i-th element corresponds to a composite matrix where only nodes' type 'i' is considered.
        # ADJ[k][i,j]=value iff an edge (i,j) exists AND node_type(i) == k
        self.CompositeAdjacencies = self.buildCompositeAdjacency()

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self):
        """ COPY METHOD

        :return: a Deep Copy of the GraphObject instance. """
        return CompositeGraphObject(arcs=self.getArcs(), nodes=self.getNodes(), targets=self.getTargets(),
                                    set_mask=self.getSetMask(), output_mask=self.getOutputMask(),
                                    sample_weights=self.getSampleWeights(), NodeGraph=self.getNodeGraph(),
                                    aggregation_mode=self.aggregation_mode, dim_node_label=self.DIM_NODE_LABEL,
                                    type_mask=self.getTypeMask())

    # -----------------------------------------------------------------------------------------------------------------
    def buildCompositeAdjacency(self):
        """ Build a list ADJ of Composite Aggregated Adjacency Matrices, s.t. ADJ[t][i,j]=value if an edge (i,j) exists AND type(i)==k
        :return: list of sparse Matrices, for memory efficiency. One for each node's type. """
        composite_adjacencies = [self.Adjacency.copy() for _ in range(len(self.DIM_NODE_LABEL))]

        for t, a in zip(self.type_mask.transpose(), composite_adjacencies):
            # set to 0 rows of nodes of incorrect type
            #type_node_mask = np.any([np.equal(self.arcs[:, 0], i) for i in np.argwhere(t)], axis=0)
            #a.data[np.logical_not(type_node_mask)] = 0
            not_type_node_mask = np.in1d(self.arcs[:, 0], np.argwhere(t), invert=True)
            a.data[not_type_node_mask] = 0
            a.eliminate_zeros()

        return composite_adjacencies

    # -----------------------------------------------------------------------------------------------------------------
    def buildArcNode(self, aggregation_mode):
        """ Build ArcNode Matrix A of shape (number_of_arcs, number_of_nodes) where A[i,j]=value if arc[i,2]==node[j].
        Compute the matmul(m:=message,A) to get the incoming message on each node, composed of nodes' states and arcs' labels.
        :return: sparse ArcNode Matrix, for memory efficiency.
        :raise: Error if <aggregation_mode> is not in ['average', 'sum', 'normalized', 'composite_average']."""
        if aggregation_mode not in ['normalized', 'average', 'sum', 'composite_average']: raise ValueError("ERROR: Unknown aggregation mode")

        # exploit super function
        if aggregation_mode in ['normalized', 'average', 'sum']:
            matrix = super().buildArcNode(aggregation_mode)

        # composite average node aggregation - incoming message as sum of averaged type-based neighbors state
        # e.g. if a node i has 3 neighbors (2 of them belonging to a type k1, the other to a type k2):
        # the message coming from k1's nodes is divided by 2,
        # while the message coming from k2's node is taken as is, being that the only one neighbor belonging to k2
        elif aggregation_mode == 'composite_average':
            # sum node aggregation - incoming message as sum of neighbors states and labels, then process composite average
            matrix = super().buildArcNode('sum')

            # set to 0 rows of nodes of incorrect type
            for t in self.type_mask.transpose():
                if not np.any(t): continue
                type_node_mask = np.any([np.equal(self.arcs[:, 0], i) for i in np.argwhere(t)], axis=0)
                val, col_index, destination_node_counts = np.unique(matrix.col[type_node_mask], return_inverse=True, return_counts=True)
                matrix.data[type_node_mask] /= destination_node_counts[col_index]

        return matrix

    # -----------------------------------------------------------------------------------------------------------------
    def setAggregation(self, aggregation_mode: str):
        """ Set ArcNode values for the specified :param aggregation_mode: """
        super().setAggregation(aggregation_mode)
        #self.ArcNode = self.buildArcNode(aggregation_mode)
        #self.Adjacency = self.buildAdjacency()
        self.CompositeAdjacencies = self.buildCompositeAdjacency()
        #self.aggregation_mode = aggregation_mode


    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self):
        """ Return a representation string of the instance of CompositeGraphObject """
        #set_mask_type = 'all' if np.all(self.set_mask) else 'mixed'
        return f"composite_{super().__repr__()}"
            #graph(n={self.nodes.shape[0]}, a={self.arcs.shape[0]}, type={self.type_mask.shape[-1]}, " \
               #f"ndim={self.DIM_NODE_LABEL.tolist()}, adim={self.DIM_ARC_LABEL}, tdim={self.DIM_TARGET}, " \
               #f"set='{set_mask_type}', mode='{self.aggregation_mode}')"

    ## GETTERS ########################################################################################################
    def getTypeMask(self):
        return self.type_mask.copy()

    ## CLASS METHODs ##################################################################################################
    @classmethod
    def get_dict_data(cls, g):
        data = super().get_dict_data(g)
        data['type_mask'] = g.type_mask
        data['dim_node_label'] = g.DIM_NODE_LABEL
        return data

     # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def merge(self, glist, problem_based: str, aggregation_mode: str, dtype='float32'):
        """ Method to merge graphs: it takes in input a list of graphs and returns them as a single graph

        :param glist: list of GraphObjects
            > NOTE if problem_based=='g', new NodeGraph will have dimension (Num nodes, Num graphs) else None
        :param aggregation_mode: str, node aggregation mode for new CompositeGraphObject, go to buildArcNode for details
        :return: a new CompositeGraphObject containing all the information (nodes, arcs, targets, etc) in glist
        """

        # get new GraphObject, then convert to CompositeGraphObject
        g = super().merge(glist, problem_based, 'sum', dtype)

        dim_node_label, type_mask = zip(*[(i.DIM_NODE_LABEL, i.getTypeMask()) for i in glist])

        # check if every graphs has the same DIM_NODE_LABEL attribute
        dim_node_label = set(tuple(i) for i in dim_node_label)
        assert len(dim_node_label) == 1, "DIM_NODE_LABEL not unique among graphs in :param glist:"

        # get single matrices for new graph
        type_mask = np.concatenate(type_mask, axis=0, dtype=bool)

        # resulting GraphObject
        return CompositeGraphObject(arcs=g.arcs, nodes=g.nodes, targets=g.targets, type_mask=type_mask,
                                    dim_node_label=dim_node_label.pop(), problem_based=problem_based,
                                    set_mask=g.set_mask, output_mask=g.output_mask, sample_weights=g.sample_weights,
                                    NodeGraph=g.NodeGraph, aggregation_mode=aggregation_mode)


    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def fromGraphTensor(self, gT, problem_based: str):
        nodegraph = coo_matrix((gT.NodeGraph.values, tf.transpose(gT.NodeGraph.indices))) if problem_based == 'g' else None
        return self(arcs=gT.arcs.numpy(), nodes=gT.nodes.numpy(), targets=gT.targets.numpy(), dim_node_label=gT.DIM_NODE_LABEL.numpy(),
                    type_mask=gT.type_mask, set_mask=gT.set_mask.numpy(), output_mask=gT.output_mask.numpy(),
                    sample_weights=gT.sample_weights.numpy(), NodeGraph=nodegraph, aggregation_mode=gT.aggregation_mode,
                    problem_based=problem_based)



#######################################################################################################################
## COMPOSITE GRAPH TENSOR CLASS #######################################################################################
#######################################################################################################################
class CompositeGraphTensor(GraphTensor):
    def __init__(self, *args, type_mask, CompositeAdjacencies, **kwargs):
        super().__init__(*args, **kwargs)

        # constant tensors + sparse tensors
        self.type_mask = tf.constant(type_mask, dtype=bool)
        self.CompositeAdjacencies = [tf.sparse.SparseTensor.from_value(i) for i in CompositeAdjacencies]

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self):
        return CompositeGraphTensor(nodes=self.nodes, dim_node_label=self.DIM_NODE_LABEL, arcs=self.arcs, targets=self.targets,
                                    set_mask=self.set_mask, output_mask=self.output_mask, sample_weights=self.sample_weights,
                                    Adjacency=self.Adjacency, ArcNode=self.ArcNode, NodeGraph=self.NodeGraph,
                                    aggregation_mode=self.aggregation_mode, type_mask=self.type_mask,
                                    CompositeAdjacencies=self.CompositeAdjacencies)

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self):
        #set_mask_type = 'all' if tf.reduce_all(self.set_mask) else 'mixed'
        return f"composite_{super().__repr__()}"
               #"graph_tensor(n={self.nodes.shape[0]}, a={self.arcs.shape[0]}, " \
               #f"type={len(self.DIM_NODE_LABEL)}, " \
               #f"ndim={self.DIM_NODE_LABEL}, adim={self.DIM_ARC_LABEL}, tdim={self.DIM_TARGET}, " \
               #f"set='{set_mask_type}', mode='{self.aggregation_mode})"

    @staticmethod
    def save_graph(graph_path: str, g, compressed: bool = False, **kwargs) -> None:
        data = {'type_mask': g.type_mask}

        for idx, mat in enumerate(g.CompositeAdjacencies):
            data[f"CompositeAdjacencies_{idx}"] = tf.concat([mat.values[:, tf.newaxis], tf.cast(mat.indices, g.dtype)], axis=1)

        super().save_graph(g, compressed, **kwargs, **data)

    @classmethod
    def load(cls, graph_npz_path, **kwargs):
        """ load a GraphTensor npz file"""
        if '.npz' not in graph_npz_path: graph_npz_path += '.npz'
        data = dict(np.load(graph_npz_path, **kwargs))

        data['aggregation_mode'] = str(data['aggregation_mode'])
        for i in ['Adjacency', 'ArcNode', 'NodeGraph']:
            data[i] = tf.SparseTensor(indices=data[i][:,1:], values=data[i][:,0], dense_shape=data.pop(i + '_shape'))

        CA = [data.pop(f"CompositeAdjacencies_{idx}") for idx, elem in enumerate(data['dim_node_label'])]
        CA = [tf.SparseTensor(indices=adj[:,1:], values=adj[:,0], dense_shape=data['Adjacency'].shape) for adj in CA]

        return cls(**data, CompositeAdjacencies=CA)

    ## CLASS and STATHIC METHODs ######################################################################################
    @classmethod
    def fromGraphObject(self, g: CompositeGraphObject):
        """ Create GraphTensor from GraphObject. Note that Adjacency and ArcNode are transposed so that GraphTensor.ArcNode and
        GraphTensor.Adjacency are ready for sparse_dense_matmul in Loop operations.
        """
        return self(nodes=g.nodes, dim_node_label=g.DIM_NODE_LABEL, arcs=g.arcs, targets=g.targets, set_mask=g.set_mask,
                    output_mask=g.output_mask, sample_weights=g.sample_weights, Adjacency=self.COO2SparseTensor(g.Adjacency),
                    ArcNode=self.COO2SparseTensor(g.ArcNode), NodeGraph=self.COO2SparseTensor(g.NodeGraph),
                    aggregation_mode=g.aggregation_mode, type_mask=g.type_mask.transpose(),
                    CompositeAdjacencies=[self.COO2SparseTensor(i) for i in g.CompositeAdjacencies])