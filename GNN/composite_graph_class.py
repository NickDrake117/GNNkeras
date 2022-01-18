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
    """ Heterogeneous Graph data representation. Composite GNNs are based on this class. """

    ## CONSTRUCTORS METHODs ###########################################################################################
    def __init__(self, nodes, arcs, targets, type_mask, dim_node_label, *args, **kwargs):
        """ CONSTRUCTOR METHOD

        :param nodes: Ordered Nodes Matrix X where nodes[i, :] = [i-th node Label].
        :param arcs: Ordered Arcs Matrix E where arcs[i, :] = [From ID Node | To ID Node | i-th arc Label].
        :param targets: Targets Matrix T with shape (Num of arcs/node targeted example or 1, dim_target example).
        :param type_mask: boolean np.array with shape (Num of nodes, Num of node's types). type_mask[:,i] refers to dim_node_label[i].
        :param dim_node_label: (list/tuple) with len == Num of node's types. i-th element defines label dimension of nodes of type i.
        :param focus: (str) The problem on which graph is used: 'a' arcs-focused, 'g' graph-focused, 'n' node-focused.
        :param set_mask: Array of boolean {0,1} to define arcs/nodes belonging to a set, when dataset == single GraphObject.
        :param output_mask: Array of boolean {0,1} to define the sub-set of arcs/nodes whose target is known.
        :param sample_weight: target sample weight for loss computation. It can be int, float or numpy.array of ints or floats:
            > If int or float, all targets are weighted as sample_weight * ones.
            > If numpy.array, len(sample_weight) and targets.shape[0] must agree.
        :param ArcNode: Sparse matrix of shape (num_of_arcs, num_of_nodes) s.t. A[i,j]=value if arc[i,2]==node[j].
        :param NodeGraph: Sparse matrix in coo format of shape (nodes.shape[0], {Num graphs or 1}) used only when focus=='g'.
        :param aggregation_mode: (str) The aggregation mode for the incoming message based on ArcNode and Adjacency matrices:
            ---> elem(matrix)={0-1};
            > 'average': A'X gives the average of incoming messages, s.t. sum(A[:,i])==1;
            > 'normalized': A'X gives the normalized message wrt the total number of g.nodes, s.t. sum(A)==1;
            > 'sum': A'X gives the total sum of incoming messages, s.t. A={0,1}.
            > 'composite_average': A'X gives the average of incoming messages wrt node's type, s.t. sum(A[:,i])>=1. """

        # type_mask[:,i] refers to nodes with DIM_NODE_LABEL[i] label dimension.
        # Be careful when initializing a new graph!
        self.type_mask = type_mask.astype(bool)

        # AFTER initializing type_mask because of self.buildAdjacency method.
        super().__init__(nodes, arcs, targets, *args, **kwargs)

        # store dimensions: first two columns of arcs contain nodes indices.
        self.DIM_NODE_LABEL = np.array(dim_node_label, ndmin=1, dtype=int)

        # build Composite Adjacency Matrices. It is a list of Adjacency Matrix as long as the number of nodes' types.
        # i-th element corresponds to a composite matrix where only nodes' type 'i' is considered.
        # ADJ[k][i,j]=value if and only if an edge (i,j) exists AND node_type(i) == k.
        self.CompositeAdjacencies = self.buildCompositeAdjacency()

    # -----------------------------------------------------------------------------------------------------------------
    def buildCompositeAdjacency(self):
        """ Build a list ADJ of Composite Aggregated Adjacency Matrices,
        s.t. ADJ[t][i,j]=value if an edge (i,j) exists AND type(i)==k.

        :return: list of sparse Matrices in coo format, for memory efficiency. One for each node's type. """
        composite_adjacencies = [self.Adjacency.copy() for _ in range(len(self.DIM_NODE_LABEL))]

        # set to 0 rows of nodes of incorrect type.
        for t, a in zip(self.type_mask.transpose(), composite_adjacencies):
            not_type_node_mask = np.in1d(self.arcs[:, 0], np.argwhere(t), invert=True)
            a.data[not_type_node_mask] = 0
            a.eliminate_zeros()

        return composite_adjacencies

    # -----------------------------------------------------------------------------------------------------------------
    def buildArcNode(self, aggregation_mode):
        """ Build ArcNode Matrix A of shape (number_of_arcs, number_of_nodes) where A[i,j]=value if arc[i,2]==node[j].
        Compute the matmul(m:=message,A) to get the incoming message on each node, composed of nodes' states and arcs' labels.

        :return: sparse ArcNode Matrix in coo format, for memory efficiency.
        :raise: Error if <aggregation_mode> is not in ['average', 'sum', 'normalized', 'composite_average']."""
        if aggregation_mode not in ['normalized', 'average', 'sum', 'composite_average']: raise ValueError("ERROR: Unknown aggregation mode")

        # initialize matrix. It's useless, just for not having any warning message at the end of the method.
        matrix = None

        # exploit super function.
        if aggregation_mode in ['normalized', 'average', 'sum']:
            matrix = super().buildArcNode(aggregation_mode)

        # composite average node aggregation - incoming message as sum of averaged type-focused neighbors state,
        # e.g. if a node i has 3 neighbors (2 of them belonging to a type k1, the other to a type k2):
        # the message coming from k1's nodes is divided by 2,
        # while the message coming from k2's node is taken as is, being that the only one neighbor belonging to k2.
        elif aggregation_mode == 'composite_average':
            # sum node aggregation - incoming message as sum of neighbors states and labels, then process composite average.
            matrix = super().buildArcNode('sum')

            # set to 0 rows of nodes of incorrect type.
            for t in self.type_mask.transpose():
                if not np.any(t): continue
                type_node_mask = np.in1d(self.arcs[:, 0], np.argwhere(t), invert=False)
                val, col_index, destination_node_counts = np.unique(matrix.col[type_node_mask], return_inverse=True, return_counts=True)
                matrix.data[type_node_mask] /= destination_node_counts[col_index]

        return matrix

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self):
        """ COPY METHOD

        :return: a Deep Copy of the GraphObject instance. """
        return CompositeGraphObject(arcs=self.getArcs(), nodes=self.getNodes(), targets=self.getTargets(),
                                    set_mask=self.getSetMask(), output_mask=self.getOutputMask(),
                                    sample_weight=self.getSampleWeights(), NodeGraph=self.getNodeGraph(),
                                    aggregation_mode=self.aggregation_mode, dim_node_label=self.DIM_NODE_LABEL,
                                    type_mask=self.getTypeMask())

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self):
        """ Representation string of the instance of CompositeGraphObject. """
        return f"composite_{super().__repr__()}"

    ## SETTERS ########################################################################################################
    def setAggregation(self, aggregation_mode: str):
        """ Set ArcNode values for the specified :param aggregation_mode: """
        super().setAggregation(aggregation_mode)
        self.CompositeAdjacencies = self.buildCompositeAdjacency()

    ## GETTERS ########################################################################################################
    # ALL return a deep copy of the corresponding element.
    def getTypeMask(self):
        return self.type_mask.copy()

    ## SAVER METHODs ##################################################################################################
    def get_dict_data(self):
        """ Return all useful elements for storing a graph :param g:, in a dict format. """
        data = super().get_dict_data()
        data['type_mask'] = self.type_mask
        data['dim_node_label'] = self.DIM_NODE_LABEL
        return data

    ## CLASS METHODs ### MERGER #######################################################################################
    @classmethod
    def merge(cls, glist, focus: str, aggregation_mode: str, dtype='float32'):
        """ Method to merge a list of CompositeGraphObject elements in a single GraphObject element.

        :param glist: list of CompositeGraphObject elements to be merged.
            > NOTE if focus=='g', new NodeGraph will have dimension (Num nodes, Num graphs).
        :param aggregation_mode: (str) incoming message aggregation mode. See BuildArcNode for details.
        :param dtype: dtype of elements of new arrays after merging procedure.
        :return: a new CompositeGraphObject containing all the information (nodes, arcs, targets, ...) in glist. """

        # get new GraphObject, then convert to CompositeGraphObject.
        g = super().merge(glist, focus, 'sum', dtype)

        dim_node_label, type_mask = zip(*[(i.DIM_NODE_LABEL, i.getTypeMask()) for i in glist])

        # check if every graphs has the same DIM_NODE_LABEL attribute.
        dim_node_label = set(tuple(i) for i in dim_node_label)
        assert len(dim_node_label) == 1, "DIM_NODE_LABEL not unique among graphs in :param glist:"

        # get single matrices for new graph.
        type_mask = np.concatenate(type_mask, axis=0, dtype=bool)

        # resulting CompositeGraphObject.
        return CompositeGraphObject(arcs=g.arcs, nodes=g.nodes, targets=g.targets, type_mask=type_mask,
                                    dim_node_label=dim_node_label.pop(), focus=focus,
                                    set_mask=g.set_mask, output_mask=g.output_mask, sample_weight=g.sample_weight,
                                    NodeGraph=g.NodeGraph, aggregation_mode=aggregation_mode)

    ## CLASS METHODs ### UTILS ########################################################################################
    @classmethod
    def fromGraphTensor(cls, g, focus: str):
        """ Create CompositeGraphObject from CompositeGraphTensor.

        :param g: a CompositeGraphTensor element to be translated into a CompositeGraphObject element.
        :param focus: (str) 'n' node-focused; 'a' arc-focused; 'g' graph-focused. See __init__ for details.
        :return: a CompositeGraphObject element whose tensor representation is g.
        """
        nodegraph = coo_matrix((g.NodeGraph.values, tf.transpose(g.NodeGraph.indices))) if focus == 'g' else None
        return cls(arcs=g.arcs.numpy(), nodes=g.nodes.numpy(), targets=g.targets.numpy(),
                   dim_node_label=g.DIM_NODE_LABEL.numpy(), type_mask=g.type_mask, set_mask=g.set_mask.numpy(),
                   output_mask=g.output_mask.numpy(), sample_weight=g.sample_weight.numpy(), NodeGraph=nodegraph,
                   aggregation_mode=g.aggregation_mode, focus=focus)


#######################################################################################################################
## COMPOSITE GRAPH TENSOR CLASS #######################################################################################
#######################################################################################################################
class CompositeGraphTensor(GraphTensor):
    """ Tensor version of a CompositeGraphObject. Useful to speed up learning processes. """

    ## CONSTRUCTORS METHODs ###########################################################################################
    def __init__(self, *args, type_mask, CompositeAdjacencies, **kwargs):
        """ It contains all information to be passed to GNN model,
        but described with tensorflow dense/sparse tensors. """
        super().__init__(*args, **kwargs)

        # constant tensors + sparse tensors.
        self.type_mask = tf.constant(type_mask, dtype=bool)
        self.CompositeAdjacencies = [tf.sparse.SparseTensor.from_value(i) for i in CompositeAdjacencies]

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self):
        """ COPY METHOD

        :return: a Deep Copy of the CompositeGraphTensor instance. """
        return CompositeGraphTensor(nodes=self.nodes, dim_node_label=self.DIM_NODE_LABEL, arcs=self.arcs,
                                    targets=self.targets, set_mask=self.set_mask, output_mask=self.output_mask,
                                    sample_weight=self.sample_weight,  Adjacency=self.Adjacency, ArcNode=self.ArcNode,
                                    NodeGraph=self.NodeGraph, aggregation_mode=self.aggregation_mode,
                                    type_mask=self.type_mask, CompositeAdjacencies=self.CompositeAdjacencies)

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self):
        """ Representation string for the instance of CompositeGraphTensor. """
        return f"composite_{super().__repr__()}"

    ## STATIC METHODs ### SAVER #######################################################################################
    @staticmethod
    def save_graph(graph_path: str, g, compressed: bool = False, **kwargs) -> None:
        """ Save a graph in a .npz compressed/uncompressed archive.

        :param graph_npz_path: path where a single .npz file will be stored, for saving the graph.
        :param g: graph of type GraphObject to be saved.
        :param compressed: bool, if True graph will be stored in a compressed npz file, npz uncompressed otherwise.
        :param kwargs: kwargs argument for for numpy.savez/numpy.savez_compressed function. """
        data = {'type_mask': g.type_mask}

        name = 'CompositeAdjacencies_'
        for idx, mat in enumerate(g.CompositeAdjacencies):
            data[f"{name}{idx}"] = tf.concat([mat.values[:, tf.newaxis], tf.cast(mat.indices, g.dtype)], axis=1)

        super().save_graph(g, compressed, **kwargs, **data)

    ## CLASS METHODs ### LOADER #######################################################################################
    @classmethod
    def load(cls, graph_npz_path, **kwargs):
        """ Load a GraphTensor from a npz compressed/uncompressed file.

        :param graph_npz_path: path to the npz graph file.
        :param kwargs: kwargs argument of numpy.load function. """
        if '.npz' not in graph_npz_path: graph_npz_path += '.npz'
        data = dict(np.load(graph_npz_path, **kwargs))

        data['aggregation_mode'] = str(data['aggregation_mode'])
        for i in ['Adjacency', 'ArcNode', 'NodeGraph']:
            data[i] = tf.SparseTensor(indices=data[i][:,1:], values=data[i][:,0], dense_shape=data.pop(i + '_shape'))

        CA = [data.pop(f"CompositeAdjacencies_{idx}") for idx, elem in enumerate(data['dim_node_label'])]
        CA = [tf.SparseTensor(indices=adj[:,1:], values=adj[:,0], dense_shape=data['Adjacency'].shape) for adj in CA]

        return cls(**data, CompositeAdjacencies=CA)

    ## CLASS and STATIC METHODs ### UTILS #############################################################################
    @classmethod
    def fromGraphObject(cls, g: CompositeGraphObject):
        """ Create CompositeGraphTensor from CompositeGraphObject.

        :param g: a CompositeGraphObject element to be translated into a CompositeGraphTensor element.
        :return: a CompositeGraphTensor element whose normal representation is g. """
        return cls(nodes=g.nodes, dim_node_label=g.DIM_NODE_LABEL, arcs=g.arcs, targets=g.targets, set_mask=g.set_mask,
                   output_mask=g.output_mask, sample_weight=g.sample_weight, Adjacency=cls.COO2SparseTensor(g.Adjacency),
                   ArcNode=cls.COO2SparseTensor(g.ArcNode), NodeGraph=cls.COO2SparseTensor(g.NodeGraph),
                   aggregation_mode=g.aggregation_mode, type_mask=g.type_mask.transpose(),
                   CompositeAdjacencies=[cls.COO2SparseTensor(i) for i in g.CompositeAdjacencies])