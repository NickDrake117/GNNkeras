import numpy as np

from GNN.composite_graph_class import CompositeGraphObject
from GNN.graph_class import GraphObject

# load numpy from txt
path = 'MUTAG_raw/'
edgesIDs = np.loadtxt(path + 'Mutagenicity_edges.txt', dtype=int, delimiter=', ')
edgesL = np.loadtxt(path + 'Mutagenicity_edge_labels.txt', dtype=int)
nodesL = np.loadtxt(path + 'Mutagenicity_node_labels.txt', dtype=int)
gIDs_nodes = np.loadtxt(path + 'Mutagenicity_graph_indicator.txt', dtype=int)
gtargs = np.loadtxt(path + 'Mutagenicity_graph_labels.txt', dtype=int)

# retrieve useful indices
_, idx = np.unique(gIDs_nodes, return_index=True)
idx = np.concatenate([idx, [len(gIDs_nodes)]])
idx = idx.tolist()

# NODES
print('Nodes', end='\t')
# encode labels in 1-hot vectors
nL = np.zeros((nodesL.shape[0], len(np.unique(nodesL))), dtype=int)
nL[range(nL.shape[0]), nodesL] = 1
nodes = [nL[i:j, :] for i, j in zip(idx[:-1], idx[1:])]
print('OK')

# EDGES
print('Edges', end='\t')
edgesIDs = np.unique(edgesIDs, axis=0)
# check edge membership in each graph
eids = [k[:, 0] * k[:, 1] for k in [(edgesIDs > i) * (edgesIDs <= j) for i, j in zip(idx[:-1], idx[1:])]]
eIDs = [edgesIDs[i, :] for i in eids]
# rename nodes indices
for i in eIDs:
    unique = np.unique(i)
    new_vals = range(len(unique))
    for k, elem in enumerate(unique): i[i == elem] = new_vals[k]
# encode labels in 1-hot vectors
eL = np.zeros((edgesL.shape[0], len(np.unique(edgesL))), dtype=int)
eL[range(eL.shape[0]), edgesL] = 1
# concatenate [id1, id2, label]
edges = [np.concatenate([eIDs[i], eL[eids[i]]], axis=1) for i in range(len(eIDs))]
print('OK')

# TARGETS
print('Targets', end='\t')
# encode labels in 1-hot vectors
targs = np.zeros((len(gtargs), len(np.unique(gtargs))), dtype=int)
targs[range(len(targs)), gtargs] = 1
print('OK')

# HOMOGENEOUS GRAPHS
graphs = [GraphObject(arcs=e, nodes=n, targets=t[np.newaxis, ...], focus='g')
          for e, n, t in zip(edges, nodes, targs)]

# HETEROGENEOUS GRAPHS
composite_graphs = [CompositeGraphObject(arcs=g.arcs, nodes=g.nodes, targets=g.targets, focus='g',
                                         type_mask=np.ones((g.nodes.shape[0], 1), dtype=bool),
                                         dim_node_features=(g.nodes.shape[1],))
                    for g in graphs]
