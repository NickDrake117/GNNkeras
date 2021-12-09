# coding=utf-8
import os
import shutil
import numpy as np
import tensorflow as tf

from scipy.sparse import coo_matrix


#######################################################################################################################
## GRAPH OBJECT CLASS #################################################################################################
#######################################################################################################################
class GraphObject:
    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self, nodes, arcs, targets,
                 problem_based: str = 'n',
                 set_mask=None,
                 output_mask=None,
                 sample_weights=1,
                 NodeGraph=None,
                 ArcNode=None,
                 aggregation_mode: str = 'sum'):
        """ CONSTRUCTOR METHOD

        :param nodes: Ordered Nodes Matrix X where nodes[i] = [Node Label].
        :param arcs: Ordered Arcs Matrix E where arcs[i] = [ID Node From | ID NodeTo | Arc Label].
        :param targets: Targets Matrix T with shape (Num of targeted example [nodes or arcs], dim_target example).
        :param problem_based: (str) The problem on which graph is used: 'a' arcs-based, 'g' graph-based, 'n' node-based.
        :param set_mask: Array of boolean {0,1} to define arcs/nodes belonging to a set, when dataset == single GraphObject.
        :param output_mask: Array of boolean {0,1} to define the sub-set of arcs/nodes whose target is known.
        :param sample_weights: target sample weight for loss computation. It can be int, float or numpy.array of ints or floats
            > If int, all targets are weighted as sample_weights * ones.
            > If numpy.array, len(sample_weights) and targets.shape[0] must agree.
        :param NodeGraph: Sparse matrix of shape (nodes.shape[0], {Num graphs or 1}) used only when problem_based=='g'.
        :param ArcNode: Sparse matrix of shape (num_of_arcs, num_of_nodes) s.t. A[i,j]=value if arc[i,2]==node[j].
        :param aggregation_mode: (str) The aggregation mode for the incoming message for a node using ArcNode and Adjacency:
            > 'average': elem(matrix)={0-1} -> A'X gives the average of incoming messages, s.t. sum(A[:,i])=1;
            > 'normalized': elem(matrix)={0-1} -> A'X gives the normalized message wrt the total number of g.nodes;
            > 'sum': elem(matrix)={0,1} -> A'X gives the total sum of incoming messages. """
        self.dtype = tf.keras.backend.floatx()

        # store arcs, nodes, targets
        self.nodes = nodes.astype(self.dtype)
        self.arcs = np.unique(arcs, axis=0).astype(self.dtype)
        self.targets = targets.astype(self.dtype)
        self.sample_weights = sample_weights * np.ones(self.targets.shape[0])

        # store dimensions: first two columns of arcs contain nodes indices
        self.DIM_NODE_LABEL = nodes.shape[1]
        self.DIM_ARC_LABEL = arcs.shape[1] - 2
        self.DIM_TARGET = targets.shape[1]

        # setting the problem type: node, arcs or graph based + check existence of passed parameters in keys
        lenMask = {'n': nodes.shape[0], 'a': arcs.shape[0], 'g': nodes.shape[0]}

        # build set_mask, for a dataset composed of only a single graph: its nodes have to be divided in Tr, Va and Te
        self.set_mask = np.ones(lenMask[problem_based], dtype=bool) if set_mask is None else set_mask.astype(bool)
        # build output_mask
        self.output_mask = np.ones(len(self.set_mask), dtype=bool) if output_mask is None else output_mask.astype(bool)

        # check lengths: output_mask must be as long as set_mask
        if len(self.set_mask) != len(self.output_mask): raise ValueError('Error - len(<set_mask>) != len(<output_mask>)')

        # nodes and arcs aggregation
        self.aggregation_mode = str(aggregation_mode)

        # build ArcNode matrix or acquire it from input
        self.ArcNode = self.buildArcNode(self.aggregation_mode) if ArcNode is None else coo_matrix(ArcNode, dtype=self.dtype)

        # build Adjancency Matrix A. Note that it can be an Aggregated Version of the 'normal' Adjacency Matrix (with only 0 and 1)
        self.Adjacency = self.buildAdjacency()

        # build node_graph conversion matrix
        self.NodeGraph = self.buildNodeGraph(problem_based) if NodeGraph is None else coo_matrix(NodeGraph, dtype=self.dtype)

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self):
        """ COPY METHOD

        :return: a Deep Copy of the GraphObject instance.
        """
        return GraphObject(nodes=self.getNodes(), arcs=self.getArcs(), targets=self.getTargets(),
                           set_mask=self.getSetMask(), output_mask=self.getOutputMask(),
                           sample_weights=self.getSampleWeights(), NodeGraph=self.getNodeGraph(),
                           aggregation_mode=self.aggregation_mode)

    # -----------------------------------------------------------------------------------------------------------------
    def buildAdjacency(self):
        """ Build the 'Aggregated' Adjacency Matrix ADJ, s.t. ADJ[i,j]=value if edge (i,j) exists in graph edges set.

        value is set by self.aggregation_mode: 'sum':1, 'normalized':1/self.nodes.shape[0], 'average':1/number_of_neighbors """
        values = self.ArcNode.data
        indices = zip(*self.arcs[:, :2].astype(int))
        return coo_matrix((values, indices), shape=(self.nodes.shape[0], self.nodes.shape[0]), dtype=self.dtype)

    # -----------------------------------------------------------------------------------------------------------------
    def buildArcNode(self, aggregation_mode):
        """ Build ArcNode Matrix A of shape (number_of_arcs, number_of_nodes) where A[i,j]=value if arc[i,2]==node[j].
        Compute the matmul(m:=message,A) to get the incoming message on each node.

        :return: sparse ArcNode Matrix, for memory efficiency.
        :raise: Error if <aggregation_mode> is not in ['average','sum','normalized']. """
        if aggregation_mode not in ['sum', 'normalized', 'average']: raise ValueError("ERROR: Unknown aggregation mode")

        col = self.arcs[:, 1]  # column indices of A are located in the second column of the arcs tensor
        row = np.arange(0, len(col))  # arc id (from 0 to number of arcs)

        # sum node aggregation
        # incoming message as sum of neighbors states and labels
        values_vector = np.ones(len(col))

        # normalized node aggregation
        # incoming message as sum of neighbors states and labels divided by the number of nodes in the graph
        if aggregation_mode == 'normalized':
            values_vector = values_vector * float(1 / len(col))

        # average node aggregation
        # incoming message as average of neighbors states and labels
        elif aggregation_mode == 'average':
            val, col_index, destination_node_counts = np.unique(col, return_inverse=True, return_counts=True)
            values_vector = values_vector / destination_node_counts[col_index]

        # isolated nodes correction: if nodes[i] is isolated, then ArcNode[:,i]=0, to maintain nodes ordering
        return coo_matrix((values_vector, (row, col)), shape=(self.arcs.shape[0], self.nodes.shape[0]), dtype=self.dtype)

    # -----------------------------------------------------------------------------------------------------------------
    def setAggregation(self, aggregation_mode: str) -> None:
        """ Set ArcNode values for the specified :param aggregation_mode: """
        self.ArcNode = self.buildArcNode(aggregation_mode)
        self.Adjacency = self.buildAdjacency()
        self.aggregation_mode = aggregation_mode

    # -----------------------------------------------------------------------------------------------------------------
    def buildNodeGraph(self, problem_based: str):
        """ Build Node-Graph Aggregation Matrix, to transform a node-based problem in a graph-based one.

        NodeGraph != empty only if problem_based == 'g': It has dimensions (nodes.shape[0], 1) for a single graph,
        or (nodes.shape[0], Num graphs) for a graph containing 2+ graphs, built by merging the single graphs into a bigger one,
        such that after the node-graph aggregation process gnn can compute (Num graphs, targets.shape[1]) as output.
        :return: non-empty NodeGraph sparse matrix if :param problem_based: is 'g', as NodeGraph is used in graph-based problems.
        """
        if problem_based == 'g': data = np.ones((self.nodes.shape[0], 1)) * 1 / self.nodes.shape[0]
        else: data = np.array([], ndmin=2)
        return coo_matrix(data, dtype=self.dtype)

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self):
        """ Return a representation string of the instance of GraphObject """
        set_mask_type = 'all' if np.all(self.set_mask) else 'mixed'
        return f"graph(n={self.nodes.shape[0]}, a={self.arcs.shape[0]}, " \
               f"ndim={self.DIM_NODE_LABEL}, adim={self.DIM_ARC_LABEL}, tdim={self.DIM_TARGET}, " \
               f"set={set_mask_type}, mode={self.aggregation_mode})"

    # -----------------------------------------------------------------------------------------------------------------
    def __str__(self):
        """ Return a representation string for the intance of GraphObject, for print() purpose. """
        return self.__repr__()

    ## GETTERS ########################################################################################################
    def getArcs(self):
        return self.arcs.copy()

    def getNodes(self):
        return self.nodes.copy()

    def getTargets(self):
        return self.targets.copy()

    def getSetMask(self):
        return self.set_mask.copy()

    def getOutputMask(self):
        return self.output_mask.copy()

    def getAdjacency(self):
        return self.Adjacency.copy()

    def getArcNode(self):
        return self.ArcNode.copy()

    def getNodeGraph(self):
        return self.NodeGraph.copy()

    def getSampleWeights(self):
        return self.sample_weights.copy()

    ## SAVER METHODs ##################################################################################################
    def save(self, graph_path: str, **kwargs) -> None:
        """ Save graph in a .npz uncompressed archive. """
        self.save_graph(graph_path, self, False, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    def save_compressed(self, graph_path, **kwargs) -> None:
        """ Save graph in a .npz compressed archive. """
        self.save_graph(graph_path, self, True, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    def savetxt(self, graph_folder_path: str, format: str = '%.10g', **kwargs) -> None:
        """ Save graph in folder. All attributes are saved in as many .txt files as needed.

        :param graph_folder_path: (str) folder path in which graph is saved.
        """
        self.save_txt(graph_folder_path, self, format, **kwargs)

    ## STATIC METHODs ### SAVER #######################################################################################
    @classmethod
    def get_dict_data(cls, g):

        # nodes, arcs and targets are saved by default
        data = {i:j for i, j in zip(['nodes', 'arcs', 'targets'], [g.nodes, g.arcs, g.targets])}

        # set_mask, output_mask and sample_weights are saved only in they have not all elements equal to 1
        if not all(g.set_mask): data['set_mask'] = g.set_mask
        if not all(g.output_mask): data['output_mask'] = g.output_mask
        if np.any(g.sample_weights != 1): data['sample_weights'] = g.sample_weights

        # NodeGraph is saved only if it is a graph_based problem and g is a merged graph
        if (g.NodeGraph.size > 0 and g.NodeGraph.shape[1] > 1):
            data['NodeGraph'] = np.stack([g.NodeGraph.data, g.NodeGraph.row, g.NodeGraph.col])

        return data

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def save_graph(graph_path: str, g, compressed: bool = False, **kwargs) -> None:
        """ Save a graph in a .npz compressed/uncompressed archive.

        :param graph_path: path where a single .npz file will be stored, for saving the graph.
        :param g: graph of type GraphObject to be saved.
        :param compressed: bool, if True graph will be stored in a compressed npz file, otherwise uncompressed
        :param kwargs: all kwargs argument for for numpy.savez/numpy.savez_compressed function
        """
        data = g.get_dict_data(g)
        saving_function = np.savez_compressed if compressed else np.savez
        saving_function(graph_path, **data, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def save_txt(graph_folder_path: str, g, fmt: str = '%.10g', **kwargs) -> None:
        """ Save a graph to a directory, creating txt files referring to all attributes of graph g
        Note that graph_folder_path will contain ONLY a single graph g. If folder is not empty, it is removed and re-created.
        Remind that dataset folder contains one folder for each graph.

        :param graph_folder_path: new directory for saving the graph.
        :param g: graph of type GraphObject to be saved.
        :param fmt: param passed to np.savetxt function.
        :param kwargs: all kwargs argument of numpy.savetxt function.
        """
        # check folder
        if graph_folder_path[-1] != '/': graph_folder_path += '/'
        if os.path.exists(graph_folder_path): shutil.rmtree(graph_folder_path)
        os.makedirs(graph_folder_path)

        data = g.get_dict_data(g)
        for i in data: np.savetxt(f"{graph_folder_path}{i}.txt", data[i], fmt=fmt, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def save_dataset(folder, glist, compressed=False, **kwargs) -> None:
        """ Save a list of GraphObjects, i.e. a dataset of graphs, in a folder. Each graph is saved as a npz file

        :param folder: (str) path for saving the dataset. If folder already exists, it is removed and re-created.
        :param glist: list of GraphObjects to be saved in the folder
        :param compressed: (bool) if True every graph is saved in a npz compressed file, otherwise npz uncompressed
        """
        if folder[-1] != '/': folder += '/'
        if os.path.exists(folder): shutil.rmtree(folder)
        os.makedirs(folder)
        for idx,g in enumerate(glist): GraphObject.save_graph(f"{folder}/g{idx}", g, compressed, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def save_dataset_txt(folder, glist, **kwargs) -> None:
        """ Save a list of GraphObjects, i.e. a dataset of graphs, in a folder. Each graph is saved as folder of txt files

        :param folder: (str) path for saving the dataset. If folder already exists, it is removed and re-created.
        :param glist: list of GraphObjects to be saved in the folder
        :param kwargs: kwargs of numpy.savetxt function
        """
        if folder[-1] != '/': folder += '/'
        if os.path.exists(folder): shutil.rmtree(folder)
        os.makedirs(folder)
        for idx,g in enumerate(glist): GraphObject.save_txt(f"{folder}/g{idx}", g, **kwargs)

    ## CLASS METHODs ### LOADER #######################################################################################
    @classmethod
    def load(cls, graph_npz_path, problem_based, aggregation_mode, **kwargs):
        """ Load a GraphObject from a npz compressed/uncompressed file
        :param graph_npz_path: path to the npz graph file
        :param problem_based: (str) : 'n'-nodeBased; 'a'-arcBased; 'g'-graphBased
        :param aggregation_mode: node aggregation mode: 'average','sum','normalized'. Go to BuildArcNode for details
        :param kwargs: kwargs of numpy.load function. """
        if '.npz' not in graph_npz_path: graph_npz_path += '.npz'
        data = dict(np.load(graph_npz_path, **kwargs))

        # Translate matrices from (length, 3) [values, index1, index2] to coo sparse matrices.
        nodegraph = data.pop('NodeGraph', None)
        if nodegraph is not None: data['NodeGraph'] = coo_matrix((nodegraph[0, :], nodegraph[1:, :].astype(int)))

        return cls(problem_based=problem_based, aggregation_mode=aggregation_mode, **data)

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def load_txt(cls, graph_folder_path: str, problem_based: str, aggregation_mode: str, **kwargs):
        """ Load a graph from a directory which contains at least 3 txt files referring to nodes, arcs and targets

        :param graph_folder_path: directory containing at least 3 files: 'nodes.txt', 'arcs.txt' and 'targets.txt'
            > other possible files: 'NodeGraph.txt','output_mask.txt' and 'set_mask.txt'. No other files required!
        :param problem_based: (str) : 'n'-nodeBased; 'a'-arcBased; 'g'-graphBased
            > NOTE  For graph_based problems, file 'NodeGraph.txt' must to be present in folder
                    NodeGraph has shape (nodes, 3) s.t. in coo_matrix NodeGraph[0,:]==data, NodeGraph[1:,:]==indices for data
        :param aggregation_mode: node aggregation mode: 'average','sum','normalized'. Go to BuildArcNode for details
        :param kwargs: kwargs of numpy.loadtxt function.
        :return: GraphObject described by files in <graph_folder_path> folder
        """
        # load all the files inside <graph_folder_path> folder
        if graph_folder_path[-1] != '/': graph_folder_path += '/'

        files = os.listdir(graph_folder_path)
        keys = [i.rsplit('.')[0] for i in files] + ['problem_based', 'aggregation_mode']
        vals = [np.loadtxt(graph_folder_path + i, ndmin=2, **kwargs) for i in files] + [problem_based, aggregation_mode]

        # create a dictionary with parameters and values to be passed to constructor and return GraphObject
        data = dict(zip(keys, vals))

        # Translate matrices from (length, 3) [values, index1, index2] to coo sparse matrices.
        nodegraph = data.pop('NodeGraph', None)
        if nodegraph is not None: data['NodeGraph'] = coo_matrix((nodegraph[:, 0], nodegraph[:, 1:].astype(int)))

        return cls(**data)

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def load_dataset(cls, folder, problem_based, aggregation_mode, **kwargs):
        """ Load a dataset of graphs stored in a folder of npz graph files. To be used after save_dataset function

        :param folder: path to the folder where npz graph files are stored
        :param problem_based: (str) : 'n'-nodeBased; 'a'-arcBased; 'g'-graphBased
        :param aggregation_mode: node aggregation mode: 'average','sum','normalized'. Go to BuildArcNode for details
        :param kwargs: kwargs of numpy.load function.
        :return: a list of GraphObjects """
        return [cls.load(f"{folder}/{g}", problem_based, aggregation_mode, **kwargs) for g in os.listdir(folder)]

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def load_dataset_txt(cls, folder, problem_based, aggregation_mode, **kwargs):
        """ Load a dataset of graphs stored in a folder of graph folders. To be used after save_dataset_txt function

        :param folder: path to the folder where graph folders are stored
        :param problem_based: (str) : 'n'-nodeBased; 'a'-arcBased; 'g'-graphBased
            > NOTE  For graph_based problems, file 'NodeGraph.txt' must to be present in folders
                    NodeGraph has shape (nodes, 3) s.t. in coo_matrix -> NodeGraph[0,:]==data, NodeGraph[1:,:]==indices for data
        :param aggregation_mode: node aggregation mode: 'average','sum','normalized'. Go to BuildArcNode for details
        :param kwargs: kwargs of numpy.loadtxt function.
        :return: a list of GraphObjects """
        return [cls.load_txt(f"{folder}/{g}", problem_based, aggregation_mode, **kwargs) for g in os.listdir(folder)]

    ## CLASS METHODs ### MERGER #######################################################################################
    @classmethod
    def merge(cls, glist: list, problem_based: str, aggregation_mode: str, dtype='float32'):
        """ Method to merge graphs: it takes in input a list of graphs and returns them as a single graph

        :param glist: list of GraphObjects to be merged
            > NOTE if problem_based=='g', new NodeGraph will have dimension (Num nodes, Num graphs) else None
        :param aggregation_mode: str, node aggregation mode for new GraphObject, go to buildArcNode for details
        :return: a new GraphObject containing all the information (nodes, arcs, targets, etc) in glist """
        get_data = lambda x: [(i.getNodes(), i.nodes.shape[0], i.getArcs(), i.getTargets(), i.getSetMask(), i.getOutputMask(),
                               i.getSampleWeights(), i.getNodeGraph()) for i in x]
        nodes, nodes_lens, arcs, targets, set_mask, output_mask, sample_weights, nodegraph_list = zip(*get_data(glist))

        # get single matrices for new graph
        for i, elem in enumerate(arcs): elem[:, :2] += sum(nodes_lens[:i])
        arcs = np.concatenate(arcs, axis=0, dtype=dtype)
        nodes = np.concatenate(nodes, axis=0, dtype=dtype)
        targets = np.concatenate(targets, axis=0, dtype=dtype)
        set_mask = np.concatenate(set_mask, axis=0, dtype=bool)
        output_mask = np.concatenate(output_mask, axis=0, dtype=bool)
        sample_weights = np.concatenate(sample_weights, axis=0, dtype=dtype)

        from scipy.sparse import block_diag
        nodegraph = block_diag(nodegraph_list, dtype=dtype)

        # resulting GraphObject
        return GraphObject(arcs=arcs, nodes=nodes, targets=targets, problem_based=problem_based,
                   set_mask=set_mask, output_mask=output_mask, sample_weights=sample_weights,
                   NodeGraph=nodegraph, aggregation_mode=aggregation_mode)

    ## CLASS METHODs ### UTILS ########################################################################################
    @classmethod
    def fromGraphTensor(cls, g, problem_based: str):
        """ Create GraphObject from GraphTensor. """
        nodegraph = coo_matrix((g.NodeGraph.values, tf.transpose(g.NodeGraph.indices))) if problem_based == 'g' else None
        return cls(arcs=g.arcs.numpy(), nodes=g.nodes.numpy(), targets=g.targets.numpy(),
                    set_mask=g.set_mask.numpy(), output_mask=g.output_mask.numpy(), sample_weights=g.sample_weights.numpy(),
                    NodeGraph=nodegraph, aggregation_mode=g.aggregation_mode, problem_based=problem_based)



#######################################################################################################################
## GRAPH TENSOR CLASS #################################################################################################
#######################################################################################################################
class GraphTensor:
    """ Tensor version of a GraphObject. Useful to speed up learning processes """

    ## CONSTRUCTORS METHODs ###########################################################################################
    def __init__(self, nodes, dim_node_label, arcs, targets, set_mask, output_mask, sample_weights,
                 Adjacency, ArcNode, NodeGraph, aggregation_mode):
        self.dtype = tf.keras.backend.floatx()
        self.aggregation_mode = aggregation_mode

        # store dimensions: first two columns of arcs contain nodes indices
        self.DIM_ARC_LABEL = arcs.shape[1] - 2
        self.DIM_TARGET = targets.shape[1]

        # constant tensors
        self.DIM_NODE_LABEL = tf.constant(dim_node_label, dtype=tf.int32)
        self.nodes = tf.constant(nodes, dtype=self.dtype)
        self.arcs = tf.constant(arcs, dtype=self.dtype)
        self.targets = tf.constant(targets, dtype=self.dtype)
        self.sample_weights = tf.constant(sample_weights, dtype=self.dtype)
        self.set_mask = tf.constant(set_mask, dtype=bool)
        self.output_mask = tf.constant(output_mask, dtype=bool)

        # sparse tensors
        self.Adjacency = tf.sparse.SparseTensor.from_value(Adjacency)
        self.ArcNode = tf.sparse.SparseTensor.from_value(ArcNode)
        self.NodeGraph = tf.sparse.SparseTensor.from_value(NodeGraph)

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self):
        """ COPY METHOD

        :return: a Deep Copy of the GraphTensor instance.
        """
        return GraphTensor(nodes=self.nodes, dim_node_label=self.nodes.shape[1], arcs=self.arcs, targets=self.targets,
                           set_mask=self.set_mask, output_mask=self.output_mask, sample_weights=self.sample_weights,
                           Adjacency=self.Adjacency, ArcNode=self.ArcNode, NodeGraph=self.NodeGraph,
                           aggregation_mode=self.aggregation_mode)

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self):
        """ a representation string for the intance of GraphTensor """
        set_mask_type = 'all' if tf.reduce_all(self.set_mask) else 'mixed'
        return f"graph_tensor(n={self.nodes.shape[0]}, a={self.arcs.shape[0]}, " \
               f"ndim={self.DIM_NODE_LABEL}, adim={self.DIM_ARC_LABEL}, tdim={self.DIM_TARGET}, " \
               f"set={set_mask_type}, mode={self.aggregation_mode}, dtype={self.dtype})"

    # -----------------------------------------------------------------------------------------------------------------
    def __str__(self):
        """ a representation string for the intance of GraphTensor, for print() purpose. """
        return self.__repr__()

    ## SAVER METHODs ##################################################################################################
    def save(self, path, **kwargs) -> None:
        """ Save graph in a .npz uncompressed archive. """
        self.save_graph(path, self, False, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    def save_compressed(self, path, **kwargs) -> None:
        """ Save graph in a .npz compressed archive. """
        self.save_graph(path, self, True, **kwargs)

    ## STATIC METHODs ### SAVER #######################################################################################
    @staticmethod
    def save_graph(graph_path: str, g, compressed: bool = False, **kwargs) -> None:

        sparse_data = {'aggregation_mode': np.array(g.aggregation_mode)}
        for i, mat in zip(['Adjacency', 'ArcNode', 'NodeGraph'], [g.Adjacency, g.ArcNode, g.NodeGraph]):
            sparse_data[i] = tf.concat([mat.values[:, tf.newaxis], tf.cast(mat.indices, g.dtype)], axis=1)
            sparse_data[i + '_shape'] = mat.shape

        saving_function = np.savez_compressed if compressed else np.savez
        saving_function(graph_path, dim_node_label=g.DIM_NODE_LABEL,
                        nodes=g.nodes, arcs=g.arcs, targets=g.targets, sample_weights=g.sample_weights,
                        set_mask=g.set_mask, output_mask=g.output_mask, **sparse_data, **kwargs)

    ## CLASS METHODs ### LOADER #######################################################################################
    @classmethod
    def load(cls, graph_npz_path, **kwargs):
        """ load a GraphTensor npz file"""
        if '.npz' not in graph_npz_path: graph_npz_path += '.npz'
        data = dict(np.load(graph_npz_path, **kwargs))

        data['aggregation_mode'] = str(data['aggregation_mode'])
        for i in ['Adjacency', 'ArcNode', 'NodeGraph']:
            data[i] = tf.SparseTensor(indices=data[i][:,1:], values=data[i][:,0], dense_shape=data.pop(i + '_shape'))

        return cls(**data)

    ## CLASS and STATIC METHODs ### UTILS #############################################################################
    @classmethod
    def fromGraphObject(self, g: GraphObject):
        """ Create GraphTensor from GraphObject. """
        return self(nodes=g.nodes, dim_node_label=g.nodes.shape[1], arcs=g.arcs, targets=g.targets,
                    set_mask=g.set_mask, output_mask=g.output_mask, sample_weights=g.sample_weights,
                    NodeGraph=self.COO2SparseTensor(g.NodeGraph), Adjacency=self.COO2SparseTensor(g.Adjacency),
                    ArcNode=self.COO2SparseTensor(g.ArcNode), aggregation_mode=g.aggregation_mode)

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def COO2SparseTensor(coo_matrix) -> tf.Tensor:
        """ Get the transposed sparse tensor from a sparse coo_matrix matrix """
        indices = np.zeros(shape=(0, 2), dtype=int)
        if coo_matrix.size > 0: indices = list(zip(coo_matrix.row, coo_matrix.col))

        # SparseTensor is created and then reordered to be correctly computable. NOTE: reorder() recommended by TF2.0+
        sparse_tensor = tf.SparseTensor(indices=indices, values=coo_matrix.data, dense_shape=coo_matrix.shape)
        sparse_tensor = tf.sparse.reorder(sparse_tensor)
        sparse_tensor = tf.cast(sparse_tensor, dtype=tf.keras.backend.floatx())
        return sparse_tensor