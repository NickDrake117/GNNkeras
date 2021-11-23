# coding=utf-8

import os
import shutil

import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix


#######################################################################################################################
## GRAPH OBJECT CLASS #################################################################################################
#######################################################################################################################
class CompositeGraphObject:
    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self, arcs, nodes, targets, type_mask, dim_node_labels,
                 problem_based: str = 'n',
                 set_mask=None,
                 output_mask=None,
                 sample_weights=1,
                 NodeGraph=None,
                 ArcNode=None,
                 aggregation_mode: str = 'average'):
        """ CONSTRUCTOR METHOD

        :param arcs: Ordered Arcs Matrix where arcs[i] = [ID Node From | ID NodeTo | Arc Label].
        :param nodes: Ordered Nodes Matrix where nodes[i] = [Node Label].
        :param targets: Targets Array with shape (Num of targeted example [nodes or arcs], dim_target example).
        :param type_mask: boolean np.array with shape (Num of nodes, Num of node's types). i-th columns refers to dim_node_labels[i]
        :param dim_node_labels: (list/tuple) with len == Num of node's types. i-th element defines label dimension of nodes of type i.
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
        self.dtype = tf.keras.backend.floatx()

        # store arcs, nodes, targets
        self.arcs = arcs.astype(self.dtype)
        self.nodes = nodes.astype(self.dtype)
        self.targets = targets.astype(self.dtype)
        self.sample_weights = sample_weights * np.ones(self.targets.shape[0])

        # store dimensions: first two columns of arcs contain nodes indices
        self.DIM_NODE_LABEL = np.array(dim_node_labels, ndmin=1, dtype=int)
        self.DIM_ARC_LABEL = arcs.shape[1] - 2
        self.DIM_TARGET = targets.shape[1]

        # type_mask[:,i] refers to nodes with DIM_NODE_LABEL[i] label dimension. Be careful when initializing a new graph
        self.type_mask = type_mask.astype(bool)

        # setting the problem type: node, arcs or graph based + check existence of passed parameters in keys
        lenMask = {'n': nodes.shape[0], 'a': arcs.shape[0], 'g': nodes.shape[0]}

        # build set_mask, for a dataset composed of only a single graph: its nodes have to be divided in Tr, Va and Te
        self.set_mask = np.ones(lenMask[problem_based], dtype=bool) if set_mask is None else set_mask.astype(bool)
        # build output_mask
        self.output_mask = np.ones(len(self.set_mask), dtype=bool) if output_mask is None else output_mask.astype(bool)

        # check lengths: output_mask must be as long as set_mask
        if len(self.set_mask) != len(self.output_mask): raise ValueError('Error - len(<set_mask>) != len(<output_mask>)')

        # nodes and arcs aggregation
        if aggregation_mode not in ['average', 'normalized', 'sum', 'composite_average']: raise ValueError('Unknown aggregation mode')
        self.aggregation_mode = aggregation_mode

        # build ArcNode matrix or acquire it from input
        self.ArcNode = self.buildArcNode() if ArcNode is None else ArcNode.astype(self.dtype)

        # build Adjancency Matrix. Note that it can be an Aggregated Version of the 'normal' Adjacency Matrix (with only 0 and 1)
        self.Adjacency = self.buildAdjacency()

        # build Composite Adjacency Matrices. It is a list of Adjacency Matrix as long as the number of nodes' types.
        # i-th element corresponds to a composite matrix where only nodes' type 'i' is considered.
        # Being ADJk the k-th element of the list, ADJk[i,j]=value iff an edge (i,j) exists AND node_type(i) == k
        self.CompositeAdjacencies = self.buildCompositeAdjacency()

        # build node_graph conversion matrix
        self.NodeGraph = self.buildNodeGraph(problem_based) if NodeGraph is None else NodeGraph.astype(self.dtype)

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self):
        """ COPY METHOD

        :return: a Deep Copy of the GraphObject instance.
        """
        return CompositeGraphObject(arcs=self.getArcs(), nodes=self.getNodes(), targets=self.getTargets(), type_mask=self.getTypeMask(),
                                    dim_node_labels=self.DIM_NODE_LABEL, set_mask=self.getSetMask(), output_mask=self.getOutputMask(),
                                    sample_weights=self.getSampleWeights(), NodeGraph=self.getNodeGraph(),
                                    aggregation_mode=self.aggregation_mode)

    # -----------------------------------------------------------------------------------------------------------------
    def buildAdjacency(self):
        """ Build 'Aggregated' Adjacency Matrix ADJ, s.t. ADJ[i,j]=value if an edge (i,j) exists in graph edges set.
        value is set by self.aggregation_mode: 'sum':1, 'normalized':1/self.nodes.shape[0], 'average':1/number_of_neighbors """
        values = self.ArcNode.data
        indices = zip(*self.arcs[:, :2].astype(int))
        return coo_matrix((values, indices), shape=(self.nodes.shape[0], self.nodes.shape[0]), dtype=self.dtype)

    def buildCompositeAdjacency(self):
        """ Build a list ADJ of Composite Aggregated Adjacency Matrices, s.t. ADJ[t][i,j]=value if an edge (i,j) exists AND type(i)==k
        :return: list of sparse Matrices, for memory efficiency. One for each node's type. """
        typed_Adjacencies = [self.Adjacency.copy() for _ in range(len(self.DIM_NODE_LABEL))]

        for t, a in zip(self.type_mask.transpose(), typed_Adjacencies):
            # set to 0 rows of nodes of incorrect type
            type_node_mask = np.any([np.equal(self.arcs[:, 0], i) for i in np.argwhere(t)], axis=0)
            a.data[np.logical_not(type_node_mask)] = 0
            a.eliminate_zeros()

        return typed_Adjacencies

    # -----------------------------------------------------------------------------------------------------------------
    def buildArcNode(self):
        """ Build ArcNode Matrix A of shape (number_of_arcs, number_of_nodes) where A[i,j]=value if arc[i,2]==node[j].
        Compute the matmul(m:=message,A) to get the incoming message on each node, composed of nodes' states and arcs' labels.
        :return: sparse ArcNode Matrix, for memory efficiency.
        :raise: Error if <aggregation_mode> is not in ['average', 'sum', 'normalized', 'composite_average'].
        """

        col = self.arcs[:, 1]  # column indices of A are located in the second column of the arcs tensor
        row = np.arange(0, len(col))  # arc id (from 0 to number of arcs)

        # sum node aggregation - incoming message as sum of neighbors states and labels
        values_vector = np.ones(len(col))

        # normalized node aggregation - incoming message as sum of neighbors states and labels divided by the number of nodes in the graph
        if self.aggregation_mode == 'normalized':
            values_vector = values_vector * float(1 / len(col))

        # average node aggregation - incoming message as average states and labels w.r.t. the of total number of neighbors
        elif self.aggregation_mode == 'average':
            val, col_index, destination_node_counts = np.unique(col, return_inverse=True, return_counts=True)
            values_vector = values_vector / destination_node_counts[col_index]

        # composite average node aggregation - incoming message as sum of averaged type-based neighbors state
        # e.g. if a node i has 3 neighbors (2 of them belonging to a type k1, the other to a type k2):
        # the message coming from k1's nodes is divided by 2,
        # while the message coming from k2's node is taken as is, being tnat the only one neighbor belonging to k2
        elif self.aggregation_mode == 'composite_average':
            # set to 0 rows of nodes of incorrect type
            for t in self.type_mask.transpose():
                if not np.any(t): continue
                type_node_mask = np.any([np.equal(self.arcs[:, 0], i) for i in np.argwhere(t)], axis=0)
                val, col_index, destination_node_counts = np.unique(col[type_node_mask], return_inverse=True, return_counts=True)
                values_vector[type_node_mask] /= destination_node_counts[col_index]

        # isolated nodes correction: if nodes[i] is isolated, then ArcNode[:,i]=0, to maintain nodes ordering
        return coo_matrix((values_vector, (row, col)), shape=(self.arcs.shape[0], self.nodes.shape[0]), dtype=self.dtype)

    # -----------------------------------------------------------------------------------------------------------------
    def setAggregation(self, aggregation_mode: str):
        """ Set ArcNode values for the specified :param aggregation_mode: """
        if aggregation_mode not in ['average', 'normalized', 'sum', 'composite_average']: raise ValueError('Unknown aggregation mode')
        self.aggregation_mode = aggregation_mode
        self.ArcNode = self.buildArcNode()
        self.Adjacency = self.buildAdjacency()
        self.CompositeAdjacencies = self.buildCompositeAdjacency()

    # -----------------------------------------------------------------------------------------------------------------
    def buildNodeGraph(self, problem_based: str):
        """ Build Node-Graph Aggregation Matrix, to transform a node-based problem in a graph-based one.
        nodegraph != None only if problem_based == 'g': It has dimensions (nodes.shape[0], 1) for a single graph, 
        or (nodes.shape[0], Num graphs) for a graph containing 2+ graphs, built by merging the single graphs into a bigger one,
        such that after the node-graph aggregation process gnn can compute (Num graphs, targets.shape[1]) as output.
        It's normalized wrt the number of nodes whose output is computed, i.e. the number of ones in output_mask.
        :return: nodegraph matrix if :param problem_based: is 'g' else None, as nodegraph is used in graph-based problems.
        """
        nodegraph = None
        if problem_based == 'g':
            nodes_output_coefficient = self.nodes.shape[0]
            nodegraph = np.ones((nodes_output_coefficient, 1), dtype=np.float32) * 1 / nodes_output_coefficient
        return nodegraph

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self):
        set_mask_type = 'all' if np.all(self.set_mask) else 'mixed'
        return f"composite_graph(n={self.nodes.shape[0]}, a={self.arcs.shape[0]}, type={self.type_mask.shape[-1]}, " \
               f"ndim={self.DIM_NODE_LABEL.tolist()}, adim={self.DIM_ARC_LABEL}, tdim={self.DIM_TARGET}, " \
               f"set='{set_mask_type}', mode='{self.aggregation_mode}')"

    # -----------------------------------------------------------------------------------------------------------------
    def __str__(self):
        return self.__repr__()

    ## SAVER METHODs ##################################################################################################
    def save(self, graph_folder_path: str) -> None:
        """ save graph in folder. All attributes are saved in numpy .npy files.

        :param graph_folder_path: (str) folder path in which graph is saved.
        """
        CompositeGraphObject.save_npy(graph_folder_path, self)

    # -----------------------------------------------------------------------------------------------------------------
    def savetxt(self, graph_folder_path: str, format: str = '%.10g') -> None:
        """ save graph in folder. All attributes are saved in textual .txt files.

        :param graph_folder_path: (str) folder path in which graph is saved.
        """
        CompositeGraphObject.save_txt(graph_folder_path, self, format)

    ## GETTERS ########################################################################################################
    def getArcs(self):
        return self.arcs.copy()

    def getNodes(self):
        return self.nodes.copy()

    def getTargets(self):
        return self.targets.copy()

    def getTypeMask(self):
        return self.type_mask.copy()

    def getSetMask(self):
        return self.set_mask.copy()

    def getOutputMask(self):
        return self.output_mask.copy()

    def getAdjacency(self):
        return self.Adjacency.copy()

    def getArcNode(self):
        return self.ArcNode.copy()

    def getNodeGraph(self):
        return None if self.NodeGraph is None else self.NodeGraph.copy()

    def getSampleWeights(self):
        return self.sample_weights.copy()

    ## CLASS METHODs ##################################################################################################
    @classmethod
    def save_npy(self, graph_folder_path: str, g):
        """ Save a graph to a directory, creating txt files referring to all attributes of graph g
        Note that graph_folder_path will contain ONLY a single graph g. If folder is not empty, it is removed and re-made
        Remind that dataset folder contains one folder for each graph.

        :param graph_folder_path: new directory for saving the graph. 
        :param g: graph of type GraphObject to be saved.
        """
        # check folder
        if graph_folder_path[-1] != '/': graph_folder_path += '/'
        if os.path.exists(graph_folder_path): shutil.rmtree(graph_folder_path)
        os.makedirs(graph_folder_path)

        # save everything
        np.save(graph_folder_path + 'arcs.npy', g.arcs)
        np.save(graph_folder_path + 'nodes.npy', g.nodes)
        np.save(graph_folder_path + 'targets.npy', g.targets)
        np.save(graph_folder_path + 'type_mask.npy', g.type_mask)
        np.save(graph_folder_path + 'dim_node_labels.npy', g.DIM_NODE_LABEL)
        if not all(g.set_mask): np.save(graph_folder_path + 'set_mask.npy', g.set_mask)
        if not all(g.output_mask): np.save(graph_folder_path + 'output_mask.npy', g.output_mask)
        if np.any(g.sample_weights != 1): np.save(graph_folder_path + 'sample_weights.npy', g.sample_weights)
        if g.NodeGraph is not None and g.targets.shape[0] > 1: np.save(graph_folder_path + 'NodeGraph.npy', g.NodeGraph)

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def save_txt(self, graph_folder_path: str, g, format: str = '%.10g'):
        """ Save a graph to a directory, creating txt files referring to all attributes of graph g
        Note that graph_folder_path will contain ONLY a single graph g. If folder is not empty, it is removed and re-made.
        Remind that dataset folder contains one folder for each graph.

        :param graph_folder_path: new directory for saving the graph.
        :param g: graph of type GraphObject to be saved.
        :param format: param passed to np.savetxt().
        """
        # check folder
        if graph_folder_path[-1] != '/': graph_folder_path += '/'
        if os.path.exists(graph_folder_path): shutil.rmtree(graph_folder_path)
        os.makedirs(graph_folder_path)

        # save everything
        np.savetxt(graph_folder_path + 'arcs.txt', g.arcs, fmt=format)
        np.savetxt(graph_folder_path + 'nodes.txt', g.nodes, fmt=format)
        np.savetxt(graph_folder_path + 'targets.txt', g.targets, fmt=format)
        np.savetxt(graph_folder_path + 'type_mask.txt', g.type_mask, fmt=format)
        if not all(g.set_mask): np.savetxt(graph_folder_path + 'set_mask.txt', g.set_mask, fmt=format)
        if not all(g.output_mask): np.savetxt(graph_folder_path + 'output_mask.txt', g.output_mask, fmt=format)
        if np.any(g.sample_weights != 1): np.savetxt(graph_folder_path + 'sample_weights.txt', g.sample_weights, fmt=format)
        if g.NodeGraph is not None and g.targets.shape[0] > 1: np.savetxt(graph_folder_path + 'NodeGraph.txt', g.NodeGraph, fmt=format)

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def load(self, graph_folder_path: str, problem_based: str, aggregation_mode: str):
        """ Load a graph from a directory which contains at least 3 numpy files referring to nodes, arcs and targets

        :param graph_folder_path: directory containing at least 3 files: 'nodes.npy', 'arcs.npy' and 'targets.npy'
            > other possible files: 'NodeGraph.npy','output_mask.npy' and 'set_mask.npy'. No other files required!
        :param aggregation_mode: node aggregation mode: 'average','sum','normalized'. Go to BuildArcNode for details
        :param problem_based: (str) : 'n'-nodeBased; 'a'-arcBased; 'g'-graphBased
            > NOTE For graph_based problems, folder must include file 'NodeGraph.npy'
        :return: GraphObject described by files in <graph_folder_path> folder
        """
        # load all the files inside <graph_folder_path> folder
        if graph_folder_path[-1] != '/': graph_folder_path += '/'
        files = os.listdir(graph_folder_path)
        keys = [i.rsplit('.')[0] for i in files] + ['problem_based', 'aggregation_mode']
        vals = [np.load(graph_folder_path + i) for i in files] + [problem_based, aggregation_mode]

        # create a dictionary with parameters and values to be passed to constructor and return GraphObject
        params = dict(zip(keys, vals))
        return self(**params)

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def load_txt(self, graph_folder_path: str, problem_based: str, aggregation_mode: str):
        """ Load a graph from a directory which contains at least 3 txt files referring to nodes, arcs and targets

        :param graph_folder_path: directory containing at least 3 files: 'nodes.txt', 'arcs.txt' and 'targets.txt'
            > other possible files: 'NodeGraph.txt','output_mask.txt' and 'set_mask.txt'. No other files required!
        :param problem_based: (str) : 'n'-nodeBased; 'a'-arcBased; 'g'-graphBased
            > NOTE For graph_based problems, folder must include file 'NodeGraph.txt'
        :param aggregation_mode: node aggregation mode: 'average','sum','normalized'. Go to BuildArcNode for details
        :return: GraphObject described by files in <graph_folder_path> folder
        """
        # load all the files inside <graph_folder_path> folder
        if graph_folder_path[-1] != '/': graph_folder_path += '/'
        files = os.listdir(graph_folder_path)
        keys = [i.rsplit('.')[0] for i in files] + ['problem_based', 'aggregation_mode']
        vals = [np.loadtxt(graph_folder_path + i, ndmin=2) for i in files] + [problem_based, aggregation_mode]

        # create a dictionary with parameters and values to be passed to constructor and return GraphObject
        params = dict(zip(keys, vals))
        return self(**params)

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def merge(self, glist, problem_based: str, aggregation_mode: str):
        """ Method to merge graphs: it takes in input a list of graphs and returns them as a single graph

        :param glist: list of GraphObjects
            > NOTE if problem_based=='g', new NodeGraph will have dimension (Num nodes, Num graphs) else None
        :param aggregation_mode: str, node aggregation mode for new GraphObject, go to buildArcNode for details
        :return: a new GraphObject containing all the information (nodes, arcs, targets, etc) in glist
        """
        # check glist parameter: others parameter are in constructor
        if not (type(glist) == list and all(isinstance(x, (CompositeGraphObject, str)) for x in glist)):
            raise TypeError('type of param <glist> must be list of str \'path-like\' or GraphObjects')

        nodes, nodes_lens, arcs, dim_node_labels, type_mask, targets, set_mask, output_mask, sample_weights, nodegraph_list = \
            zip(*[(i.getNodes(), i.nodes.shape[0], i.getArcs(), i.DIM_NODE_LABEL, i.getTypeMask(), i.getTargets(),
                   i.getSetMask(), i.getOutputMask(), i.getSampleWeights(), i.getNodeGraph()) for i in glist])

        # check if every graphs has the same DIM_NODE_LABEL attribute
        dim_node_labels = set(tuple(i) for i in dim_node_labels)
        assert len(dim_node_labels) == 1

        # get single matrices for new graph
        for i, elem in enumerate(arcs): elem[:, :2] += sum(nodes_lens[:i])
        arcs = np.concatenate(arcs, axis=0)
        nodes = np.concatenate(nodes, axis=0)
        targets = np.concatenate(targets, axis=0)
        type_mask = np.concatenate(type_mask, axis=0)
        dim_node_labels = dim_node_labels.pop()
        set_mask = np.concatenate(set_mask, axis=0)
        output_mask = np.concatenate(output_mask, axis=0)
        sample_weights = np.concatenate(sample_weights, axis=0)

        nodegraph = None
        if problem_based == 'g':
            from scipy.linalg import block_diag
            nodegraph = block_diag(*nodegraph_list)

        # resulting GraphObject
        return self(arcs=arcs, nodes=nodes, targets=targets, type_mask=type_mask, dim_node_labels=dim_node_labels,
                    problem_based=problem_based, set_mask=set_mask, output_mask=output_mask, sample_weights=sample_weights,
                    NodeGraph=nodegraph, aggregation_mode=aggregation_mode)

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def fromGraphTensor(self, gT, problem_based: str):
        nodegraph = None
        if problem_based == 'g': nodegraph = gT.NodeGraph.numpy()
        return self(arcs=gT.arcs.numpy(), nodes=gT.nodes.numpy(), targets=gT.targets.numpy(), dim_node_labels=gT.DIM_NODE_LABELS.numpy(),
                    set_mask=gT.set_mask.numpy(), output_mask=gT.output_mask.numpy(), sample_weights=gT.sample_weights.numpy(),
                    NodeGraph=nodegraph, aggregation_mode=gT.aggregation_mode, problem_based=problem_based)


class CompositeGraphTensor:
    def __init__(self, nodes, arcs, targets, type_mask, dim_node_labels, set_mask, output_mask, sample_weights, Adjacency,
                 CompositeAdjacencies, ArcNode, NodeGraph, aggregation_mode):
        dtype = tf.keras.backend.floatx()

        # store dimensions: first two columns of arcs contain nodes indices
        self.DIM_NODE_LABEL = tf.constant(dim_node_labels, dtype=tf.int32)
        self.DIM_ARC_LABEL = arcs.shape[1] - 2
        self.DIM_TARGET = targets.shape[1]

        self.nodes = tf.constant(nodes, dtype=dtype)
        self.arcs = tf.constant(arcs, dtype=dtype)
        self.targets = tf.constant(targets, dtype=dtype)
        self.sample_weights = tf.constant(sample_weights, dtype=dtype)
        self.type_mask = tf.constant(type_mask, dtype=bool)
        self.set_mask = tf.constant(set_mask, dtype=bool)
        self.output_mask = tf.constant(output_mask, dtype=bool)
        self.DIM_NODE_LABELS = tf.constant(dim_node_labels)
        self.aggregation_mode = aggregation_mode

        self.NodeGraph = tf.constant([], dtype=dtype)
        if NodeGraph is not None: self.NodeGraph = tf.constant(NodeGraph, dtype=dtype)

        # Adjacency and ArcNode in GraphTensor MUST BE already transposed!
        self.Adjacency = tf.sparse.SparseTensor.from_value(Adjacency)
        self.CompositeAdjacencies = [tf.sparse.SparseTensor.from_value(i) for i in CompositeAdjacencies]
        self.ArcNode = tf.sparse.SparseTensor.from_value(ArcNode)

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self):
        return CompositeGraphTensor(nodes=self.nodes, arcs=self.arcs, targets=self.targets, type_mask=self.type_mask,
                                    dim_node_labels=self.DIM_NODE_LABELS, set_mask=self.set_mask,
                                    output_mask=self.output_mask, sample_weights=self.sample_weights,
                                    Adjacency=self.Adjacency, CompositeAdjacencies=self.CompositeAdjacencies,
                                    ArcNode=self.ArcNode, NodeGraph=self.NodeGraph,
                                    aggregation_mode=self.aggregation_mode)

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self):
        set_mask_type = 'all' if tf.reduce_all(self.set_mask) else 'mixed'
        return f"composite_graph_tensor(n={self.nodes.shape[0]}, a={self.arcs.shape[0]}, " \
               f"type={len(self.DIM_NODE_LABEL)}, " \
               f"ndim={self.DIM_NODE_LABEL}, adim={self.DIM_ARC_LABEL}, tdim={self.DIM_TARGET}, " \
               f"set='{set_mask_type}', mode='{self.aggregation_mode})"

    # -----------------------------------------------------------------------------------------------------------------
    def __str__(self):
        return self.__repr__()

    ## CLASS and STATHIC METHODs ######################################################################################
    @classmethod
    def fromGraphObject(self, g: CompositeGraphObject):
        """ Create GraphTensor from GraphObject. Note that Adjacency and ArcNode are transposed so that GraphTensor.ArcNode and
        GraphTensor.Adjacency are ready for sparse_dense_matmul in Loop operations.
        """
        return self(nodes=g.nodes, arcs=g.arcs, targets=g.targets, type_mask=g.type_mask.transpose(),
                    dim_node_labels=g.DIM_NODE_LABEL, set_mask=g.set_mask, output_mask=g.output_mask,
                    sample_weights=g.sample_weights, NodeGraph=g.NodeGraph,
                    Adjacency=self.COO2SparseTransposedTensor(g.Adjacency),
                    CompositeAdjacencies=[self.COO2SparseTransposedTensor(i) for i in g.CompositeAdjacencies],
                    ArcNode=self.COO2SparseTransposedTensor(g.ArcNode), aggregation_mode=g.aggregation_mode)

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def COO2SparseTransposedTensor(coo_matrix) -> tf.Tensor:
        """ Get the transposed sparse tensor from a sparse coo_matrix matrix """
        # SparseTensor is created and then reordered to be correctly computable. NOTE: reorder() recommended by TF2.0+
        indices = list(zip(coo_matrix.col, coo_matrix.row))
        if not indices: indices = np.empty(shape=(0, 2), dtype=int)
        sparse_tensor = tf.SparseTensor(indices, values=coo_matrix.data, dense_shape=[coo_matrix.shape[1], coo_matrix.shape[0]])
        sparse_tensor = tf.sparse.reorder(sparse_tensor)
        sparse_tensor = tf.cast(sparse_tensor, dtype=tf.float32)
        return sparse_tensor
