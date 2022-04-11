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
    """ Homogeneous Graph data representation. Non-Composite GNNs are based on this class. """

    ## CONSTRUCTORS METHODs ###########################################################################################
    def __init__(self, nodes, arcs, targets,
                 focus: str = 'n',
                 set_mask=None,
                 output_mask=None,
                 sample_weight=1,
                 NodeGraph=None,
                 aggregation_mode: str = 'sum',
                 dtype='float32') -> None:
        """ CONSTRUCTOR METHOD

        :param nodes: Ordered Nodes Matrix X where nodes[i, :] = [i-th node Label].
        :param arcs: Ordered Arcs Matrix E where arcs[i, :] = [From ID Node | To ID Node | i-th arc Label].
                     Note that [From ID Node | To ID Node] are used only for building Adjacency Matrix.
                     Note alse that a bidirectional edge must be described as 2 arcs [i,j, arc_features] and [j, i, arc_features].
                     Edge matrices is composed of only edge features.
        :param targets: Targets Matrix T with shape (Num of arcs/nodes/graphs targeted examples).
        :param focus: (str) The problem on which graph is used: 'a' arcs-focused, 'g' graph-focused, 'n' node-focused.
        :param set_mask: Array of boolean {0,1} to define arcs/nodes belonging to a set, useful when dataset == single GraphObject.
        :param output_mask: Array of boolean {0,1} to define the sub-set of arcs/nodes whose target is known.
                            For graph-focuse proble, it is automatically set to 1 for very nodes.
        :param sample_weight: target sample weight for loss computation. It can be int, float or numpy.array of ints or floats:
            > If int or float, all targets are weighted as sample_weight * ones.
            > If numpy.array, len(sample_weight) and targets.shape[0] must agree.
        :param NodeGraph: Sparse scipy matrix in coo format of shape (nodes.shape[0], {Num graphs or 1}) used only when focus=='g'.
        :param aggregation_mode: (str) The aggregation mode for the incoming message based on ArcNode and Adjacency matrices:
            ---> elem(matrix)={0-1}; Deafult to 'sum'.
            > 'average': A'X gives the average of incoming messages, s.t. sum(A[:,i])==1;
            > 'normalized': A'X gives the normalized message wrt the total number of g.nodes, s.t. sum(A)==1;
            > 'sum': A'X gives the total sum of incoming messages, s.t. A={0,1}.
        :param dtype: set dtype for dense and sparse matrices. Default to float32. """
        self._dtype = np.dtype(dtype)

        # store arcs, nodes, targets and sample weight and dimensions.
        # Note that first two columns of arcs contain nodes indices, BUT they're not integrated in arcs matrix
        arcs = np.unique(arcs, axis=0)
        self.arcs = arcs[:, 2:].astype(self._dtype)
        self.nodes = nodes.astype(self._dtype)
        self.targets = targets.astype(self._dtype)
        self.sample_weight = (sample_weight * np.ones(self.targets.shape[0])).astype(self._dtype)

        # setting the problem the graph is focused on: node, arcs or graph focused.
        lenMask = {'n': nodes.shape[0], 'a': arcs.shape[0], 'g': nodes.shape[0]}

        # build set_mask and output mask
        # for a dataset composed of only a single graph, its nodes must be divided into training, test and validation set.
        self.set_mask = np.ones(lenMask[focus], dtype=bool) if set_mask is None else set_mask.astype(bool)
        self.output_mask = np.ones(lenMask[focus], dtype=bool) if output_mask is None else output_mask.astype(bool)

        # check lengths: output_mask must be as long as set_mask, if passed as parameter to constructor.
        if len(self.set_mask) != len(self.output_mask): raise ValueError('Error - len(<set_mask>) != len(<output_mask>)')

        # set nodes and arcs aggregation.
        self.checkAggregation(aggregation_mode)
        self._aggregation_mode = str(aggregation_mode)

        # build Adjancency Matrix A.
        # Note that it may be an aggregated version of the 'normal' Adjacency Matrix (with only 0 and 1),
        # since each element is set from aggregation mode.
        self.Adjacency = self.buildAdjacency(arcs[:, :2])

        # build ArcNode matrix or set from constructor's parameter.
        self.ArcNode = self.buildArcNode()

        # build node_graph conversion matrix, to transform a node-focused output into a graph-focused one.
        self.NodeGraph = self.buildNodeGraph(focus) if NodeGraph is None else coo_matrix(NodeGraph, dtype=self._dtype)

    # -----------------------------------------------------------------------------------------------------------------
    def buildAdjacency(self, indices):
        """ Build the 'Aggregated' Adjacency Matrix ADJ, s.t. ADJ[i,j]=value if edge (i,j) exists in E.

        Note that if the edge is bidirection, both (i,j) and (j,i) must exist in indices.
        Values are set by self.aggregation_mode: 'sum':1, 'normalized':1/self.nodes.shape[0], 'average':1/number_of_neighbors.

        :return: sparse ArcNode Matrix in coo format, for memory efficiency. """

        # column indices of A are located in the second column of the arcs tensor,
        # since they represent the node id pointed by the corresponding arc.
        # row == arc id, is an ordered array of int from 0 to number of arcs.
        indices = indices.astype(int)
        col = indices[:, 1]

        # SUM node aggregation
        # incoming message as sum of neighbors states and labels.
        # ADJ in {0, 1}.
        values = np.ones(len(col))

        # NORMALIZED node aggregation
        # incoming message as sum of neighbors states and labels divided by the number of nodes in the graph.
        # sum(ADJ)==1.
        if self.aggregation_mode == 'normalized':
            values = values * float(1 / len(col))

        # AVERAGE node aggregation
        # incoming message as average of neighbors states and labels.
        # sum(ADJ[:, i])==1.
        elif self.aggregation_mode == 'average':
            val, col_index, destination_node_counts = np.unique(col, return_inverse=True, return_counts=True)
            values = values / destination_node_counts[col_index]

        # isolated nodes correction: if nodes[i] is isolated, then ADJ[:,i]=0, to maintain nodes ordering.
        return coo_matrix((values, (indices[:, 0], col)), shape=(self.nodes.shape[0], self.nodes.shape[0]), dtype=self.dtype)

    # -----------------------------------------------------------------------------------------------------------------
    def buildArcNode(self):
        """ Build ArcNode Matrix AN of shape (number_of_arcs, number_of_nodes) where A[i,j]=value if arc[i, 1]==j-th node.
        Compute the matmul(A, msg:=message) to get the incoming message on each node.

        :return: sparse ArcNode Matrix in coo format, for memory efficiency. """
        values = self.Adjacency.data
        indices = (np.arange(0, len(values)), self.Adjacency.col)

        # isolated nodes correction: if nodes[i] is isolated, then AN[:,i]=0, to maintain nodes ordering.
        return coo_matrix((values, indices), shape=(self.arcs.shape[0], self.nodes.shape[0]), dtype=self.dtype)

    # -----------------------------------------------------------------------------------------------------------------
    def buildNodeGraph(self, focus: str):
        """ Build Node-Graph Aggregation Matrix, to transform a node-focused gnn output in a graph-focused one.

        NodeGraph != empty only if focus == 'g': It has dimensions (nodes.shape[0], 1) for a single graph,
        or (nodes.shape[0], Num graphs) for a graph containing 2+ graphs, built by merging the single graphs into a bigger one,
        such that after the node-graph aggregation process gnn can compute (Num graphs, targets.shape[1]) as output.

        :return: empty or non-empty NodeGraph sparse matrix in coo_format: if :param focus: is 'g', as NodeGraph
        is used in graph-focused problems. """
        if focus == 'g':
            data = np.ones((self.nodes.shape[0], 1)) * (1 / self.nodes.shape[0])
        else:
            data = np.array([], ndmin=2)
        return coo_matrix(data, dtype=self.dtype)

    ## COPY METHODs ###################################################################################################
    def copy(self):
        """ COPY METHOD

        :return: a Deep Copy of the GraphObject instance. """
        return self.from_config(self.get_config())

    # -----------------------------------------------------------------------------------------------------------------
    def __copy__(self):
        """ Copy inline method """
        return self.copy()

    # -----------------------------------------------------------------------------------------------------------------
    def __deepcopy__(self):
        """ Deep copy inline method """
        return self.copy()

    ## OPERATORS METHODs ##############################################################################################
    def __eq__(self, g) -> bool:
        """ Equality inline method. Check equality between graphs, in lines like g1 == g2. """
        # check representation strings, as they are immediate to obtain and summarize graphs in the same way
        if str(self) != str(g): return False

        # if strings are the same, then check for all the quantities
        # Note that Adjacency and ArcNode are not checked in GraphObjects,
        # since they are calculated from arcs indices and aggregation_mode
        self_config, g_config = self.get_config(savedata=True), g.get_config(savedata=True)
        for key in self_config:
            if not np.array_equal(self_config[key], g_config[key]): return False

        # everything matches -> self and g are equals
        return True

    ## CONFIG METHODs #################################################################################################
    def get_config(self, savedata: bool = False) -> dict:
        """ Return all useful elements for storing a graph :param g:, in a dict format. """

        # nodes, arcs, targets are saved by default.
        # set_mask, output_mask are saved by default since they define the problem the graph is focused on
        # arcs matrix are built so that each row contains [idx node outgoing arc, idx node ingoing arc, arc's features]
        config = {i: j for i, j in zip(['nodes', 'arcs', 'targets', 'set_mask', 'output_mask'],
                                       [self.nodes, self._get_indexed_arcs(), self.targets, self.set_mask, self.output_mask])}

        # sample_weight are saved only in they have not all elements equal to 1.
        if np.any(self.sample_weight != 1): config['sample_weight'] = self.sample_weight

        # if save data, nodegraph is saved in dense form [row, col, value]
        if any(self.NodeGraph.data):
            if savedata: config['NodeGraph'] = np.stack([self.NodeGraph.row, self.NodeGraph.col, self.NodeGraph.data]).transpose()
            else: config['NodeGraph'] = self.NodeGraph

        return config

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        """ Create class from configuration dictionary. To be used with get_config().
        It is good practice providing this method to user. """
        return cls(**config)

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self) -> str:
        """ Representation string of the instance of GraphObject. """
        return f"graph(n={self.nodes.shape[0]}:{self.DIM_NODE_FEATURES}, " \
               f"a={self.arcs.shape[0]}:{self.DIM_ARC_FEATURES}, " \
               f"t={self.targets.shape[0]}:{self.DIM_TARGET}, " \
               f"set={'all' if np.all(self.set_mask) else 'mixed'}, " \
               f"mode={self.aggregation_mode}, dtype={self.dtype.name})"

    # -----------------------------------------------------------------------------------------------------------------
    def __str__(self) -> str:
        """ Representation string for the instance of GraphObject, for print() purpose. """
        return self.__repr__()

    ## STATIC METHODs ### UTILS #######################################################################################
    @staticmethod
    def checkAggregation(aggregation_mode):
        """ Check aggregation_mode parameter. Must be in ['average', 'sum', 'normalized'].

        :raise: Error if :param aggregation_mode: is not in ['average', 'sum', 'normalized']."""
        if str(aggregation_mode) not in ['sum', 'normalized', 'average']:
            raise ValueError("ERROR: Unknown aggregation mode")

    ## PROPERTY GETTERS ###############################################################################################
    @property
    def nodes(self):
        return self._nodes

    @property
    def arcs(self):
        return self._arcs

    @property
    def targets(self):
        return self._targets

    @property
    def DIM_NODE_FEATURES(self):
        return self._DIM_NODE_FEATURES

    @property
    def DIM_ARC_FEATURES(self):
        return self._DIM_ARC_FEATURES

    @property
    def DIM_TARGET(self):
        return self._DIM_TARGET

    @property
    def aggregation_mode(self):
        return self._aggregation_mode

    @property
    def dtype(self):
        return self._dtype

    ## PROPERTY SETTERS ###############################################################################################
    @nodes.setter
    def nodes(self, nodes):
        """ Set node features matrix X, then automatically change DIM_NODE_FEATURES attribute """
        self._nodes = nodes
        self._DIM_NODE_FEATURES = np.array(nodes.shape[1], ndmin=1, dtype=int)

    @arcs.setter
    def arcs(self, arcs):
        """ Set arc features matrix E, then automatically change DIM_ARC_FEATURES attribute """
        self._arcs = arcs
        self._DIM_ARC_FEATURES = arcs.shape[1]

    @targets.setter
    def targets(self, targets):
        """ Set target matrix T, then automatically change DIM_TARGET attribute """
        self._targets = targets
        self._DIM_TARGET = targets.shape[1]

    @aggregation_mode.setter
    def aggregation_mode(self, aggregation_mode: str):
        """ Set ArcNode values for the specified :param aggregation_mode: """
        self.checkAggregation(aggregation_mode)
        self._aggregation_mode = str(aggregation_mode)
        self.Adjacency = self.buildAdjacency(self._get_indexed_arcs()[:, :2])
        self.ArcNode = self.buildArcNode()

    @dtype.setter
    def dtype(self, dtype='float32'):
        """ Cast GraphObject variables to :param dtype: dtype. """
        self._dtype = np.dtype(dtype)
        self.nodes = self.nodes.astype(dtype)
        self.arcs = self.arcs.astype(dtype)
        self.targets = self.targets.astype(dtype)
        self.sample_weight = self.sample_weight.astype(dtype)
        self.Adjacency = self.Adjacency.astype(dtype)
        self.ArcNode = self.ArcNode.astype(dtype)
        self.NodeGraph = self.NodeGraph.astype(dtype)

    ## GETTERS ########################################################################################################
    def _get_indexed_arcs(self):
        return np.hstack([np.array((self.Adjacency.row, self.Adjacency.col)).transpose(), self.arcs])

    ## SAVER METHODs ##################################################################################################
    def save(self, graph_npz_path: str, **kwargs) -> None:
        """ Save graph in a .npz uncompressed archive.

        :param graph_npz_path: (str) path in which graph is saved.
        :param kwargs: kwargs argument of np.save function. """
        self.save_graph(graph_npz_path, self, False, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    def save_compressed(self, graph_npz_path, **kwargs) -> None:
        """ Save graph in a .npz compressed archive.

        :param graph_npz_path: (str) path in which graph is saved.
        :param kwargs: kwargs argument of np.savez function. """
        self.save_graph(graph_npz_path, self, True, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    def savetxt(self, graph_folder_path: str, format: str = '%.10g', **kwargs) -> None:
        """ Save graph in folder. All attributes are saved in as many .txt files as needed.

        :param graph_folder_path: (str) path in which graph is saved.
        :param kwargs: kwargs argument of np.savetxt function.
        """
        self.save_txt(graph_folder_path, self, format, **kwargs)

    ## STATIC METHODs ### SAVER #######################################################################################
    @staticmethod
    def save_graph(graph_npz_path: str, g, compressed: bool = False, **kwargs) -> None:
        """ Save a graph in a .npz compressed/uncompressed archive.

        :param graph_npz_path: path where a single .npz file will be stored, for saving the graph.
        :param g: graph of type GraphObject to be saved.
        :param compressed: bool, if True graph will be stored in a compressed npz file, npz uncompressed otherwise.
        :param kwargs: kwargs argument for for numpy.savez/numpy.savez_compressed function. """
        config = g.get_config(savedata=True)
        saving_function = np.savez_compressed if compressed else np.savez
        saving_function(graph_npz_path, **config, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def save_txt(graph_folder_path: str, g, fmt: str = '%.10g', **kwargs) -> None:
        """ Save a graph to a directory, creating txt files referring to all needed attributes of graph g

        Note that graph_folder_path will contain ONLY a single graph g.
        If folder is not empty, it is removed and re-created.
        Remind that a generic dataset folder contains one folder for each graph.

        :param graph_folder_path: directory for saving the graph.
        :param g: graph of type GraphObject to be saved.
        :param fmt: param format passed to np.savetxt function.
        :param kwargs: kwargs argument of numpy.savetxt function. """

        # check folder.
        if graph_folder_path[-1] != '/': graph_folder_path += '/'
        if os.path.exists(graph_folder_path): shutil.rmtree(graph_folder_path)
        os.makedirs(graph_folder_path)
        config = g.get_config(savedata=True)
        for i in config: np.savetxt(f"{graph_folder_path}{i}.txt", config[i], fmt=fmt, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def save_dataset(folder, glist, compressed=False, **kwargs) -> None:
        """ Save a dataset of graphs, in the form of a list of GraphObjects, in a folder.
        Each graph is saved as a npz file. Then the dataset is a folder of npz files.

        :param folder: (str) path for saving the dataset. If folder already exists, it is removed and re-created.
        :param glist: list of GraphObjects to be saved in the folder.
        :param compressed: (bool) if True every graph is saved in a npz compressed file, npz uncompressed otherwise.
        :param kwargs: keargs argument of np.save/np.savez functions. """
        if folder[-1] != '/': folder += '/'
        if os.path.exists(folder): shutil.rmtree(folder)
        os.makedirs(folder)
        for idx, g in enumerate(glist): GraphObject.save_graph(f"{folder}/g{idx}", g, compressed, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def save_dataset_txt(folder, glist, **kwargs) -> None:
        """ Save a dataset of graphs, in the form of a list of GraphObjects, in a folder.
         Each graph is saved as folder of txt files. Then the dataset is a folder of folders of txt files.

        :param folder: (str) path for saving the dataset. If folder already exists, it is removed and re-created.
        :param glist: list of GraphObjects to be saved in the folder.
        :param kwargs: kwargs argument of numpy.savetxt function. """
        if folder[-1] != '/': folder += '/'
        if os.path.exists(folder): shutil.rmtree(folder)
        os.makedirs(folder)
        for idx, g in enumerate(glist): GraphObject.save_txt(f"{folder}/g{idx}", g, **kwargs)

    ## CLASS METHODs ### LOADER #######################################################################################
    @classmethod
    def load(cls, graph_npz_path, focus, **kwargs):
        """ Load a GraphObject from a npz compressed/uncompressed file.

        :param graph_npz_path: path to the npz graph file.
        :param focus: (str) 'n' node-focused; 'a' arc-focused; 'g' graph-focused. See __init__ for details.
        :param kwargs: kwargs argument of numpy.load function + ['dtype', 'aggregation_mode']. See Constructor for details.
        :return: a GraphObject element. """
        if '.npz' not in graph_npz_path: graph_npz_path += '.npz'

        aggregation_mode = kwargs.pop('aggregation_mode', 'sum')
        dtype = kwargs.pop('dtype', 'float32')
        data = dict(np.load(graph_npz_path, **kwargs))

        # Translate matrices from (length, 3) [index1, index2, values] to coo sparse matrices.
        if focus=='g' and np.any(data.get('NodeGraph', list())):
            data['NodeGraph'] = coo_matrix((data['NodeGraph'][:, 2], zip(*data['NodeGraph'][:, :2].astype(int))))

        return cls(focus=focus, aggregation_mode=aggregation_mode, dtype=dtype, **data)

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def load_txt(cls, graph_folder_path: str, focus: str, **kwargs):
        """ Load a graph from a directory which contains at least 3 txt files referring to nodes, arcs and targets.

        :param graph_folder_path: directory containing at least 3 files: 'nodes.txt', 'arcs.txt' and 'targets.txt'
            > other possible files: 'NodeGraph.txt', 'output_mask.txt', 'set_mask.txt' and 'sample_weight.txt'.
            No other files required!
        :param focus: (str) 'n' node-focused; 'a' arc-focused; 'g' graph-focused. See __init__ for details.
            > NOTE  For graph-focused problems, file 'NodeGraph.txt' must to be present in folder.
            NodeGraph has shape (nodes, 3) s.t. in coo_matrix NodeGraph[:, 0]==data and NodeGraph[:, 1:]==indices for data.
        :param kwargs: kwargs argument of numpy.loadtxt function + ['dtype', 'aggregation_mode']. See Constructor for details.
        :return: GraphObject described by files in <graph_folder_path> folder. """

        # load all the files inside <graph_folder_path> folder.
        if graph_folder_path[-1] != '/': graph_folder_path += '/'

        files = os.listdir(graph_folder_path)
        dtype = kwargs.pop('dtype', 'float32')
        aggregation_mode = kwargs.pop('aggregation_mode', 'sum')

        # create a dictionary with parameters and values to be passed to GraphObject's constructor.
        keys = [i.rsplit('.')[0] for i in files]
        vals = [np.loadtxt(graph_folder_path + i, ndmin=2, **kwargs) for i in files]
        data = dict(zip(keys, vals))

        # Translate matrices from (length, 3) [index1, index2, values] to coo sparse matrices.
        if focus=='g' and np.any(data.get('NodeGraph', list())):
            data['NodeGraph'] = coo_matrix((data['NodeGraph'][:, 2], zip(*data['NodeGraph'][:, :2].astype(int))))

        return cls(focus=focus, aggregation_mode=aggregation_mode, dtype=dtype, **data)

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def load_dataset(cls, folder, focus, **kwargs) -> list:
        """ Load a dataset of graphs stored in a folder of npz graph files.
        To be used after save_dataset method.

        :param folder: path to the folder where npz graph files are stored.
        :param focus: (str) 'n' node-focused; 'a' arc-focused; 'g' graph-focused. See __init__ for details.
        :param kwargs: kwargs argument of numpy.load and GraphObject.load functions.
        :return: a list of GraphObject elements. """
        return [cls.load(f"{folder}/{g}", focus, **kwargs) for g in os.listdir(folder)]

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def load_dataset_txt(cls, folder, focus, **kwargs) -> list:
        """ Load a dataset of graphs stored in a folder of graph folders.
        To be used after save_dataset_txt method.

        :param folder: path to the folder where graph folders are stored.
        :param focus: (str) 'n' node-focused; 'a' arc-focused; 'g' graph-focused. See __init__ for details.
            > NOTE  For graph-focused problems, file 'NodeGraph.txt' must to be present in folders.
            NodeGraph has shape (nodes, 3) s.t. in coo_matrix NodeGraph[:, 0]==data and NodeGraph[:, 1:]==indices for data.
        :param kwargs: kwargs argument of numpy.loadtxt and GraphObject.load functions.
        :return: a list of GraphObject elements. """
        return [cls.load_txt(f"{folder}/{g}", focus, **kwargs) for g in os.listdir(folder)]

    ## NORMALIZERS ####################################################################################################
    def normalize(self, scalers: dict[dict], return_scalers: bool = True, apply_on_graph: bool = True):
        """ Normalize GraphObject with an arbitrary scaler. Work well tith scikit-learn preprocessing scalers.

        :param scalers: (dict). Possible keys are ['nodes', 'arcs', 'targets']
                        scalers[key] is a dict with possible keys in ['class', 'kwargs']
                        scalers[key]['class'] is the scaler class of the arbitrary scaler
                        scalers[key]['kwargs'] are the keywords for fitting the arbitrary scaler on key data.
        :param return_scalers: (bool). If True, a dictionary scaler_dict is returned. Default True.
                               The output is a dict with possible keys in [nodes, arcs, targets].
                               If a scaler is missing, related key is not used.
                               For example, if scalers_kwargs.keys() in [['nodes','targets'], ['targets','nodes']],
                               the output is ad dict {'nodes': nodes_scaler, 'targets': target_scaler}.
        :param apply_on_graph: (bool). If True, scalers are applied on self data;
                               If False, self data is used only to get scalers params,
                               but no normalization is applied afterwards. """

        # output scaler, if needed
        scalers_output_dict = dict()

        for key in scalers:
            if key not in ['nodes', 'arcs', 'targets']: raise KeyError("KEY not recognised in :param scalers: in graph.normalize()")

        # nodes
        if 'nodes' in scalers:
            node_scaler = scalers['nodes']['class'](**scalers['nodes'].get('kwargs', dict())).fit(self.nodes)
            scalers_output_dict['nodes'] = node_scaler
            if apply_on_graph: self.nodes = node_scaler.transform(self.nodes)

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
        if 'nodes' in scalers: self.nodes = scalers['nodes'].transform(self.nodes)

        # normalize arcs if arcs features are available
        if 'arcs' in scalers and self.DIM_ARC_FEATURES > 0: self.arcs = scalers['arcs'].transform(self.arcs)

        # normalize targets
        if 'targets' in scalers: self.targets = scalers['targets'].transform(self.targets)

    ## CLASS METHODs ### MERGER #######################################################################################
    @classmethod
    def merge(cls, glist: list, focus: str, aggregation_mode: str, dtype='float32'):
        """ Method to merge a list of GraphObject elements in a single GraphObject element.

        :param glist: list of GraphObject elements to be merged.
            > NOTE if focus=='g', new NodeGraph will have dimension (Num nodes, Num graphs).
        :param aggregation_mode: (str) incoming message aggregation mode. See BuildArcNode for details.
        :param dtype: dtype of elements of new arrays after merging procedure.
        :return: a new GraphObject containing all the information (nodes, arcs, targets, ...) in glist. """
        get_data = lambda x: [(i.nodes, i.nodes.shape[0], i._get_indexed_arcs(),
                               i.targets, i.set_mask, i.output_mask,
                               i.sample_weight, i.NodeGraph) for i in x]
        nodes, nodes_lens, arcs, targets, set_mask, output_mask, sample_weight, nodegraph_list = zip(*get_data(glist))

        # get single matrices for new graph
        nodes = np.concatenate(nodes, axis=0, dtype=dtype)
        for i, elem in enumerate(arcs): elem[:, :2] += sum(nodes_lens[:i])
        arcs = np.concatenate(arcs, axis=0, dtype=dtype)
        targets = np.concatenate(targets, axis=0, dtype=dtype)
        set_mask = np.concatenate(set_mask, axis=0, dtype=bool)
        output_mask = np.concatenate(output_mask, axis=0, dtype=bool)
        sample_weight = np.concatenate(sample_weight, axis=0, dtype=dtype)

        from scipy.sparse import block_diag
        nodegraph = block_diag(nodegraph_list, dtype=dtype)

        # resulting GraphObject.
        return GraphObject(nodes=nodes, arcs=arcs, targets=targets, focus=focus,
                           set_mask=set_mask, output_mask=output_mask, sample_weight=sample_weight,
                           NodeGraph=nodegraph, aggregation_mode=aggregation_mode, dtype=dtype)

    ## CLASS METHODs ### UTILS ########################################################################################
    @classmethod
    def fromGraphTensor(cls, g, focus: str, dtype='float32'):
        """ Create GraphObject from GraphTensor.

        :param g: a GraphTensor element to be translated into a GraphObject element.
        :param focus: (str) 'n' node-focused; 'a' arc-focused; 'g' graph-focused. See __init__ for details.
        :return: a GraphObject element whose tensor representation is g.
        """
        nodegraph = coo_matrix((g.NodeGraph.values, tf.transpose(g.NodeGraph.indices))) if focus == 'g' else None
        return cls(nodes=g.nodes.numpy(), arcs=np.hstack([g.Adjacency.indices, g.arcs.numpy()]), targets=g.targets.numpy(),
                   set_mask=g.set_mask.numpy(), output_mask=g.output_mask.numpy(), sample_weight=g.sample_weight.numpy(),
                   NodeGraph=nodegraph, aggregation_mode=g.aggregation_mode, focus=focus, dtype=dtype)


#######################################################################################################################
## GRAPH TENSOR CLASS #################################################################################################
#######################################################################################################################
class GraphTensor:
    """ Tensor version of a GraphObject. Useful to speed up learning processes. """
    _aggregation_dict = {'sum': 0, 'normalize': 1, 'average': 2}

    ## CONSTRUCTORS METHODs ###########################################################################################
    def __init__(self, nodes, dim_node_features, arcs, targets, set_mask, output_mask, sample_weight,
                 Adjacency, ArcNode, NodeGraph, aggregation_mode, dtype='float32') -> None:
        """ It contains all information to be passed to GNN model,
        but described with tensorflow dense/sparse tensors. """

        self.dtype = tf.as_dtype(dtype)
        self.aggregation_mode = str(aggregation_mode)

        # store dimensions: first two columns of arcs contain nodes indices.
        self.DIM_ARC_FEATURES = arcs.shape[1]
        self.DIM_TARGET = targets.shape[1]

        # constant dense tensors.
        self.DIM_NODE_FEATURES = tf.constant(dim_node_features, dtype='int32')
        self.nodes = tf.constant(nodes, dtype=self.dtype)
        self.arcs = tf.constant(arcs, dtype=self.dtype)
        self.targets = tf.constant(targets, dtype=self.dtype)
        self.sample_weight = tf.constant(sample_weight, dtype=self.dtype)
        self.set_mask = tf.constant(set_mask, dtype=bool)
        self.output_mask = tf.constant(output_mask, dtype=bool)

        # sparse tensors.
        self.Adjacency = tf.sparse.SparseTensor.from_value(Adjacency)
        self.ArcNode = tf.sparse.SparseTensor.from_value(ArcNode)
        self.NodeGraph = tf.sparse.SparseTensor.from_value(NodeGraph)

    ## COPY METHODs ###################################################################################################
    copy = GraphObject.copy
    __copy__ = GraphObject.__copy__
    __deepcopy__ = GraphObject.__deepcopy__

    ## OPERATORS METHODs ##############################################################################################
    __eq__ = GraphObject.__eq__

    ## CONFIG METHODs #################################################################################################
    def get_config(self, savedata: bool = False) -> dict:
        config = {'nodes': self.nodes, 'dim_node_features': self.DIM_NODE_FEATURES, 'arcs': self.arcs, 'targets': self.targets,
                  'set_mask': self.set_mask, 'output_mask': self.output_mask, 'sample_weight': self.sample_weight,
                  'Adjacency': self.Adjacency, 'ArcNode': self.ArcNode, 'NodeGraph': self.NodeGraph,
                  'aggregation_mode': self.aggregation_mode}

        if savedata:
            aggregation_int = self._aggregation_dict.get(self.aggregation_mode)
            config['aggregation_mode'] = np.array(aggregation_int, dtype=int)
            for i, mat in zip(['Adjacency', 'ArcNode', 'NodeGraph'], [self.Adjacency, self.ArcNode, self.NodeGraph]):
                config[i] = tf.concat([tf.cast(mat.indices, self.dtype), mat.values[:, tf.newaxis]], axis=1)
                config[i + '_shape'] = mat.shape
        return config

    # -----------------------------------------------------------------------------------------------------------------
    from_config = classmethod(GraphObject.from_config)

    ## REPRESENTATION METHODs #########################################################################################
    def __repr__(self):
        """ Representation string for the instance of GraphTensor """
        repr = GraphObject.__repr__(self).split('(',1)
        return f"{repr[0]}_tensor({repr[1]}"

    # -----------------------------------------------------------------------------------------------------------------
    __str__ = GraphObject.__str__

    ## SAVER METHODs ##################################################################################################
    save = GraphObject.save
    save_compressed = GraphObject.save_compressed
    save_graph = staticmethod(GraphObject.save_graph)
    save_dataset = staticmethod(GraphObject.save_dataset)

    ## LOADER METHODs #################################################################################################
    @classmethod
    def load(cls, graph_npz_path, **kwargs):
        """ Load a GraphTensor from a npz compressed/uncompressed file.

        :param graph_npz_path: path to the npz graph file.
        :param kwargs: kwargs argument of numpy.load function. + ['dtype'] """
        if '.npz' not in graph_npz_path: graph_npz_path += '.npz'
        dtype = kwargs.pop('dtype', 'float32')
        data = dict(np.load(graph_npz_path, **kwargs))

        # aggregation mode
        aggregation_dict = cls._aggregation_dict
        data['aggregation_mode'] = dict(zip(aggregation_dict.values(), aggregation_dict.keys()))[int(data['aggregation_mode'])]

        # sparse matrices
        for i in ['Adjacency', 'ArcNode', 'NodeGraph']:
            data[i] = tf.SparseTensor(indices=data[i][:, :2], values=data[i][:, 2], dense_shape=data.pop(i + '_shape'))

        return cls(**data, dtype=dtype)

    # -----------------------------------------------------------------------------------------------------------------
    load_dataset = classmethod(GraphObject.load_dataset)

    ## CLASS and STATIC METHODs ### UTILS #############################################################################
    @classmethod
    def fromGraphObject(cls, g: GraphObject):
        """ Create GraphTensor from GraphObject.

        :param g: a GraphObject element to be translated into a GraphTensor element.
        :return: a GraphTensor element whose normal representation is g. """
        return cls(nodes=g.nodes, dim_node_features=g.DIM_NODE_FEATURES, arcs=g.arcs, targets=g.targets,
                   set_mask=g.set_mask, output_mask=g.output_mask, sample_weight=g.sample_weight,
                   NodeGraph=cls.COO2SparseTensor(g.NodeGraph), Adjacency=cls.COO2SparseTensor(g.Adjacency),
                   ArcNode=cls.COO2SparseTensor(g.ArcNode), aggregation_mode=g.aggregation_mode, dtype=g.dtype)

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def COO2SparseTensor(coo_matrix) -> tf.Tensor:
        """ Get the sparse tensor from a sparse :param coo_matrix: matrix. """
        indices = np.zeros(shape=(0, 2), dtype=int)
        if coo_matrix.size > 0: indices = list(zip(coo_matrix.row, coo_matrix.col))

        # SparseTensor is created and then reordered to be correctly computable. NOTE: reorder() recommended by TF2.0+.
        sparse_tensor = tf.SparseTensor(indices=indices, values=coo_matrix.data, dense_shape=coo_matrix.shape)
        sparse_tensor = tf.sparse.reorder(sparse_tensor)
        sparse_tensor = tf.cast(sparse_tensor, dtype=coo_matrix.dtype)
        return sparse_tensor
