# coding=utf-8

import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix

from GNN.graph_class import GraphObject, GraphTensor


#######################################################################################################################
## COMPOSITE GRAPH OBJECT CLASS #######################################################################################
#######################################################################################################################
class CompositeGraphObject(GraphObject):
    """ Heterogeneous Graph data representation. Composite GNNs are based on this class. """
    _aggregation_dict = {'sum': 0, 'normalize': 1, 'average': 2, 'composite_average': 3}

    ## CONSTRUCTORS METHODs ###########################################################################################
    def __init__(self, nodes, arcs, targets, type_mask, dim_node_features, *args, **kwargs) -> None:
        """ CONSTRUCTOR METHOD

        :param nodes: Ordered Nodes Matrix X where nodes[i, :] = [i-th node Label].
        :param arcs: Ordered Arcs Matrix E where arcs[i, :] = [From ID Node | To ID Node | i-th arc Label].
                     Note that [From ID Node | To ID Node] are used only for building Adjacency Matrix.
                     Note alse that a bidirectional edge must be described as 2 arcs [i,j, arc_features] and [j, i, arc_features].
                     Edge matrices is composed of only edge features.
        :param targets: Targets Matrix T with shape (Num of arcs/nodes/graphs targeted examples).
        :param type_mask: boolean np.array with shape (Num of nodes, Num of node's types). type_mask[:,i] refers to dim_node_features[i].
        :param dim_node_features: (list/tuple/1D array) with len == Num of node's types.
                                i-th element defines label dimension of nodes of type i.
        :param focus: (str) The problem on which graph is used: 'a' arcs-focused, 'g' graph-focused, 'n' node-focused.
        :param set_mask: Array of boolean {0,1} to define arcs/nodes belonging to a set, when dataset == single GraphObject.
        :param output_mask: Array of boolean {0,1} to define the sub-set of arcs/nodes whose target is known.
        :param sample_weight: target sample weight for loss computation. It can be int, float or numpy.array of ints or floats:
            > If int or float, all targets are weighted as sample_weight * ones.
            > If numpy.array, len(sample_weight) and targets.shape[0] must agree.
        :param NodeGraph: Sparse matrix in coo format of shape (nodes.shape[0], {Num graphs or 1}) used only when focus=='g'.
        :param aggregation_mode: (str) The aggregation mode for the incoming message based on ArcNode and Adjacency matrices:
            ---> elem(matrix)={0-1}; Deafult to 'sum'.
            > 'average': A'X gives the average of incoming messages, s.t. sum(A[:,i])==1;
            > 'normalized': A'X gives the normalized message wrt the total number of g.nodes, s.t. sum(A)==1;
            > 'sum': A'X gives the total sum of incoming messages, s.t. A={0,1}.
            > 'composite_average': A'X gives the average of incoming messages wrt node's type, s.t. sum(A[:,i])>=1. """

        # type_mask[:,i] refers to nodes with DIM_NODE_FEATURES[i] label dimension.
        # Be careful when initializing a new graph!
        # BEFORE super().__init__() because of self.buildAdjacency() which is called-back in super().__init__(...)
        self.type_mask = type_mask.astype(bool)

        # AFTER initializing type_mask because of self.buildAdjacency() which is called-back in super().__init__(...)
        super().__init__(nodes, arcs, targets, *args, **kwargs)
        self._DIM_NODE_FEATURES = np.array(np.array(dim_node_features).squeeze(), ndmin=1, dtype=int)

        # build Composite Adjacency Matrices. It is a list of Adjacency Matrix as long as the number of nodes' types.
        # i-th element corresponds to a composite matrix where only nodes' type 'i' is considered.
        # ADJ[k][i,j]=value if and only if an edge (i,j) exists AND node_type(i) == k.
        self.CompositeAdjacencies = self.buildCompositeAdjacency()

    # -----------------------------------------------------------------------------------------------------------------
    def buildAdjacency(self, indices):
        """ Build ArcNode Matrix A of shape (number_of_arcs, number_of_nodes) where A[i,j]=value if arc[i,2]==node[j].
        Compute the matmul(m:=message,A) to get the incoming message on each node, composed of nodes' states and arcs' labels.

        :return: sparse ArcNode Matrix in coo format, for memory efficiency. """

        # initialize matrix. It's useless, just for not having any warning message at the end of the method.
        matrix = None
        indices = indices.astype(int)

        # exploit super function.
        if self.aggregation_mode in ['normalized', 'average', 'sum']:
            matrix = super().buildAdjacency(indices)

        # composite average node aggregation - incoming message as sum of averaged type-focused neighbors state,
        # e.g. if a node i has 3 neighbors (2 of them belonging to a type k1, the other to a type k2):
        # the message coming from k1's nodes is divided by 2,
        # while the message coming from k2's node is taken as is, being that the only one neighbor belonging to k2.
        elif self.aggregation_mode == 'composite_average':
            # sum node aggregation - incoming message as sum of neighbors states and labels, then process composite average.
            # Since in super buildAdjacency, no check on aggregation-mode is performed: if it is not correct, 'sum' is performed.
            matrix = super().buildAdjacency(indices)

            # set to 0 rows of nodes of incorrect type.
            for t in self.type_mask.transpose():
                if not np.any(t): continue
                type_node_mask = np.in1d(matrix.row, np.argwhere(t), invert=False)
                val, col_index, destination_node_counts = np.unique(matrix.col[type_node_mask], return_inverse=True, return_counts=True)
                matrix.data[type_node_mask] /= destination_node_counts[col_index]

        return matrix

    # -----------------------------------------------------------------------------------------------------------------
    def buildCompositeAdjacency(self):
        """ Build a list ADJ of Composite Aggregated Adjacency Matrices,
        s.t. ADJ[t][i,j]=value if an edge (i,j) exists AND type(i)==k.

        :return: list of sparse Matrices in coo format, for memory efficiency. One for each node's type. """
        composite_adjacencies = [self.Adjacency.copy() for _ in range(len(self.DIM_NODE_FEATURES))]

        # set to 0 rows of nodes of incorrect type.
        for t, a in zip(self.type_mask.transpose(), composite_adjacencies):
            not_type_node_mask = np.in1d(a.row, np.argwhere(t), invert=True)
            a.data[not_type_node_mask] = 0
            a.eliminate_zeros()

        return composite_adjacencies

    ## CONFIG METHODs #################################################################################################
    def get_config(self, savedata: bool = False) -> dict:
        """ Return all useful elements for storing a graph :param g:, in a dict format. """
        config = super().get_config(savedata)
        config['type_mask'] = self.type_mask
        config['dim_node_features'] = self.DIM_NODE_FEATURES
        return config

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self) -> str:
        """ Representation string of the instance of CompositeGraphObject. """
        return f"composite_{super().__repr__()}"

    ## PROPERTY GETTERS ###############################################################################################
    @property
    def nodes(self):
        return super().nodes

    @property
    def aggregation_mode(self):
        return super().aggregation_mode

    @property
    def dtype(self):
        return super().dtype

    ## PROPERTY SETTERS ###############################################################################################
    @nodes.setter
    def nodes(self, nodes):
        # No DIM NODE FEATURES update is performed on Composite Graphs
        self._nodes = nodes

    @aggregation_mode.setter
    def aggregation_mode(self, aggregation_mode: str):
        """ Set ArcNode values for the specified :param aggregation_mode: """
        super(CompositeGraphObject, type(self)).aggregation_mode.fset(self, aggregation_mode)
        self.CompositeAdjacencies = self.buildCompositeAdjacency()

    @dtype.setter
    def dtype(self, dtype='float32'):
        """ Cast CompositeGraphObject variables to :param dtype: dtype. """
        super(CompositeGraphObject, type(self)).dtype.fset(self, dtype)
        self.CompositeAdjacencies = [i.astype(dtype) for i in self.CompositeAdjacencies]

    ## STATIC METHODs ### UTILS #######################################################################################
    @staticmethod
    def checkAggregation(aggregation_mode):
        """ Check aggregation_mode parameter. Must be in ['average', 'sum', 'normalized', 'composite_average'].

        :raise: Error if :param aggregation_mode: is not in ['average', 'sum', 'normalized', 'composite_average']."""
        if aggregation_mode not in ['sum', 'normalized', 'average', 'composite_average']:
            raise ValueError("ERROR: Unknown aggregation mode")

    ## NORMALIZERS ####################################################################################################
    def normalize(self, scalers: dict[dict], return_scalers: bool = True, apply_on_graph: bool = True):
        """ Normalize GraphObject with an arbitrary scaler. Work well tith scikit-learn preprocessing scalers.

        :param scalers: (dict). Possible keys are ['nodes', 'arcs', 'targets']
                        scalers[key] is a dict with possible keys in ['class', 'kwargs']
                        scalers[key]['class'] is the scaler class of the arbitrary scaler
                        scalers[key]['kwargs'] are the keywords for fitting the arbitrary scaler on key data.
        :param return_scalers: (bool). If True, a dictionary scaler_dict is returned.
                               The output is a dict with possible keys in [nodes, arcs, targets].
                               If a scaler is missing, related key is not used.
                               For example, if scalers_kwargs.keys() in [['nodes','targets'], ['targets','nodes']],
                               the output is ad dict {'nodes': nodes_scaler, 'targets': target_scaler}.
        :param apply_on_graph: (bool). If True, scalers are applied on self data;
                               If False, self data is used only to get scalers params,
                               but no normalization is applied afterwards. """

        # output scaler, if needed
        scalers_output_dict = dict()

        # nodes
        if 'nodes' in scalers:
            scalers_output_dict['nodes'] = dict()
            # dim node features is considered to prevent padding values to be scaled, too.
            for idx, (mask, dim) in enumerate(zip(self.type_mask.transpose(), self.DIM_NODE_FEATURES)):
                node_scaler = scalers['nodes']['class'](**scalers['nodes'].get('kwargs', dict())).fit(self.nodes[mask, :dim])
                scalers_output_dict['nodes'][idx] = node_scaler
                if apply_on_graph: self.nodes[mask, :dim] = node_scaler.transform(self.nodes[mask, :dim])

        # arcs if arcs features are available
        if 'arcs' in scalers and self.DIM_ARC_FEATURES > 0:
            arc_scaler = scalers['arcs']['class'](**scalers['arcs'].get('kwargs', dict())).fit(self.arcs)
            scalers_output_dict['arcs'] = arc_scaler
            if apply_on_graph: self.arcs = arc_scaler.transform(self.arcs)

        # targets
        if 'targets' in scalers:
            target_scaler = scalers['targets']['class'](**scalers['targets'].get('kwargs', dict())).fit(self.targets)
            scalers_output_dict['targets'] = target_scaler
            if apply_on_graph: self.targets = target_scaler.transform(self.targets)

        if return_scalers:
            return scalers_output_dict

    # -----------------------------------------------------------------------------------------------------------------
    def normalize_from(self, scalers: dict[dict]):
        # normalize nodes
        if 'nodes' in scalers:
            for idx, (mask, dim) in enumerate(zip(self.type_mask.transpose(), self.DIM_NODE_FEATURES)):
                self.nodes[mask, :dim] = scalers['nodes'][idx].transform(self.nodes[mask, :dim])

        # normalize arcs if arcs features are available
        if 'arcs' in scalers and self.DIM_ARC_FEATURES > 0: self.arcs = scalers['arcs'].transform(self.arcs)

        # normalize targets
        if 'targets' in scalers: self.targets = scalers['targets'].transform(self.targets)

    ## CLASS METHODs ### MERGER #######################################################################################
    @classmethod
    def merge(cls, glist: list, focus: str, aggregation_mode: str, dtype='float32'):
        """ Method to merge a list of CompositeGraphObject elements in a single GraphObject element.

        :param glist: list of CompositeGraphObject elements to be merged.
            > NOTE if focus=='g', new NodeGraph will have dimension (Num nodes, Num graphs).
        :param aggregation_mode: (str) incoming message aggregation mode. See BuildArcNode for details.
        :param dtype: dtype of elements of new arrays after merging procedure.
        :return: a new CompositeGraphObject containing all the information (nodes, arcs, targets, ...) in glist. """

        # get new GraphObject, then convert to CompositeGraphObject.
        g = super().merge(glist, focus, 'sum', dtype)

        dim_node_features, type_mask = zip(*[(i.DIM_NODE_FEATURES, i.type_mask) for i in glist])

        # check if every graphs has the same DIM_NODE_FEATURES attribute.
        dim_node_features = set(tuple(i) for i in dim_node_features)
        assert len(dim_node_features) == 1, "DIM_NODE_FEATURES not unique among graphs in :param glist:"

        # get single matrices for new graph.
        type_mask = np.concatenate(type_mask, axis=0, dtype=bool)

        # resulting CompositeGraphObject.
        return CompositeGraphObject(nodes=g.nodes, arcs=g._get_indexed_arcs(), targets=g.targets,
                                    type_mask=type_mask, dim_node_features=dim_node_features.pop(), focus=focus,
                                    set_mask=g.set_mask, output_mask=g.output_mask, sample_weight=g.sample_weight,
                                    NodeGraph=g.NodeGraph, aggregation_mode=aggregation_mode, dtype=dtype)

    ## CLASS METHODs ### UTILS ########################################################################################
    @classmethod
    def fromGraphTensor(cls, g, focus: str, dtype='float32'):
        """ Create CompositeGraphObject from CompositeGraphTensor.

        :param g: a CompositeGraphTensor element to be translated into a CompositeGraphObject element.
        :param focus: (str) 'n' node-focused; 'a' arc-focused; 'g' graph-focused. See __init__ for details.
        :return: a CompositeGraphObject element whose tensor representation is g.
        """
        nodegraph = coo_matrix((g.NodeGraph.values, tf.transpose(g.NodeGraph.indices))) if focus == 'g' else None
        return cls(nodes=g.nodes.numpy(), arcs=np.hstack([g.Adjacency.indices, g.arcs.numpy()]), targets=g.targets.numpy(),
                   dim_node_features=g.DIM_NODE_FEATURES.numpy(), type_mask=g.type_mask, set_mask=g.set_mask.numpy(),
                   output_mask=g.output_mask.numpy(), sample_weight=g.sample_weight.numpy(), NodeGraph=nodegraph,
                   aggregation_mode=g.aggregation_mode, focus=focus, dtype=dtype)


#######################################################################################################################
## COMPOSITE GRAPH TENSOR CLASS #######################################################################################
#######################################################################################################################
class CompositeGraphTensor(GraphTensor):
    """ Tensor version of a CompositeGraphObject. Useful to speed up learning processes. """
    _aggregation_dict = {'sum': 0, 'normalize': 1, 'average': 2, 'composite_average': 3}

    ## CONSTRUCTORS METHODs ###########################################################################################
    def __init__(self, *args, type_mask, CompositeAdjacencies, **kwargs) -> None:
        """ It contains all information to be passed to GNN model,
        but described with tensorflow dense/sparse tensors. """
        super().__init__(*args, **kwargs)

        # constant tensors + sparse tensors.
        self.type_mask = tf.constant(type_mask, dtype=bool)
        self.CompositeAdjacencies = [tf.sparse.SparseTensor.from_value(i) for i in CompositeAdjacencies]

    ## CONFIG METHODs #################################################################################################
    def get_config(self, savedata: bool = False) -> dict:
        config = super().get_config(savedata)
        config['dim_node_features'] = self.DIM_NODE_FEATURES
        config['type_mask'] = self.type_mask
        return config

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self) -> str:
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
        data = {f"CompositeAdjacencies_{idx}": tf.concat([tf.cast(mat.indices, g.dtype), mat.values[:, tf.newaxis]], axis=1)
                for idx, mat in enumerate(g.CompositeAdjacencies)}

        GraphTensor.save_graph(graph_path, g, compressed, **kwargs, **data)

    ## CLASS METHODs ### LOADER #######################################################################################
    @classmethod
    def load(cls, graph_npz_path, **kwargs):
        """ Load a GraphTensor from a npz compressed/uncompressed file.

        :param graph_npz_path: path to the npz graph file.
        :param kwargs: kwargs argument of numpy.load function. """
        if '.npz' not in graph_npz_path: graph_npz_path += '.npz'
        dtype = kwargs.pop('dtype', 'float32')
        data = dict(np.load(graph_npz_path, **kwargs))

        # aggregation mode
        aggregation_dict = cls._aggregation_dict
        data['aggregation_mode'] = dict(zip(aggregation_dict.values(), aggregation_dict.keys()))[int(data['aggregation_mode'])]

        # sparse matrices
        for i in ['Adjacency', 'ArcNode', 'NodeGraph']:
            data[i] = tf.SparseTensor(indices=data[i][:, :2], values=data[i][:, 2], dense_shape=data.pop(i + '_shape'))

        CA = [data.pop(f"CompositeAdjacencies_{idx}") for idx, _ in enumerate(data['dim_node_features'])]
        CA = [tf.SparseTensor(indices=adj[:, :2], values=adj[:, 2], dense_shape=data['Adjacency'].shape) for adj in CA]

        return cls(**data, CompositeAdjacencies=CA, dtype=dtype)

    ## CLASS and STATIC METHODs ### UTILS #############################################################################
    @classmethod
    def fromGraphObject(cls, g: CompositeGraphObject):
        """ Create CompositeGraphTensor from CompositeGraphObject.

        :param g: a CompositeGraphObject element to be translated into a CompositeGraphTensor element.
        :return: a CompositeGraphTensor element whose normal representation is g. """
        return cls(nodes=g.nodes, dim_node_features=g.DIM_NODE_FEATURES, arcs=g.arcs, targets=g.targets, set_mask=g.set_mask,
                   output_mask=g.output_mask, sample_weight=g.sample_weight, Adjacency=cls.COO2SparseTensor(g.Adjacency),
                   ArcNode=cls.COO2SparseTensor(g.ArcNode), NodeGraph=cls.COO2SparseTensor(g.NodeGraph),
                   aggregation_mode=g.aggregation_mode, type_mask=g.type_mask.transpose(),
                   CompositeAdjacencies=[cls.COO2SparseTensor(i) for i in g.CompositeAdjacencies], dtype=g.dtype)
