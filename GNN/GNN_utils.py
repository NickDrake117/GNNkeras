from typing import Union, Optional

import numpy as np

from GNN.Sequencers.GraphSequencers import MultiGraphSequencer, SingleGraphSequencer, CompositeMultiGraphSequencer, CompositeSingleGraphSequencer
from GNN.graph_class import GraphObject


# ---------------------------------------------------------------------------------------------------------------------
def simple_graph(focus: str, aggregation_mode: str = 'average') -> GraphObject:
    """ return a single simple GraphObject for debugging purpose """
    nodes = np.array([[11, 21], [12, 22], [13, 23], [14, 24]])
    arcs = np.array([[0, 1, 10], [0, 2, 40], [1, 0, 10], [1, 2, 20], [2, 0, 40], [2, 1, 20], [2, 3, 30], [3, 2, 30]])

    tn = {'n': nodes.shape[0], 'a': arcs.shape[0], 'g': 1}
    target_number = tn[focus]
    targs = np.zeros((target_number, 2))

    if focus in ['a', 'n']:
        from sklearn.cluster import AgglomerativeClustering
        j = AgglomerativeClustering(n_clusters=2).fit(arcs[:, 2:] if focus == 'a' else nodes)
        i = range(target_number)
        targs[i, j.labels_] = 1
    else:
        targs[0, 1] = 1

    return GraphObject(arcs=arcs, nodes=nodes, targets=targs, focus=focus, aggregation_mode=aggregation_mode)

# ---------------------------------------------------------------------------------------------------------------------
def prepare_LKO_data(dataset: Union[GraphObject, list[GraphObject], list[list[GraphObject]]],
                     number_of_batches: int = 10, useVa: bool = False, seed: Optional[float] = None) -> dict:
    """ Prepare dataset for Leave K-Out procedure. The output of this function must be passed as arg[0] of model.LKO() method.
    :param dataset: a single GraphObject OR a list of GraphObject OR list of lists of GraphObject on which <gnn> has to be valuated
                    > NOTE: for graph-based problem, if type(dataset) == list of GraphObject,
                    s.t. len(dataset) == number of graphs in the dataset, then i-th class will may be have different frequencies among batches
                    [so the i-th class may me more present in a batch and absent in another batch].
                    Otherwise, if type(dataset) == list of lists, s.t. len(dataset) == number of classes AND len(dataset[i]) == number of graphs
                    belonging to i-th class, then i-th class will have the same frequency among all the batches
                    [so the i-th class will be as frequent in a single batch as in the entire dataset].
    :param number_of_batches: (int) define how many batches will be considered in LKO procedure.
    :param useVa: (bool) if True, Early Stopping is considered during learning procedure; None otherwise.
    :param seed: (int or None) for fixed-shuffle options."""
    # define useful lambda function to be used in any case
    flatten = lambda l: [item for sublist in l for item in sublist]
    assert number_of_batches > 1 + useVa, "number of batches must be grater than 1; greater than 2 if validation is used."

    # Shuffling procedure: set or not seed parameter, then shuffle classes and/or elements in each class/dataset
    if seed: np.random.seed(seed)

    # define lists for LKO -> output
    output_dict = {i: list() for i in ['training', 'test', 'validation']}

    ### SINGLE GRAPH CASE: batches are obtaind by setting set_masks for training, test and validation (if any)
    if isinstance(dataset, GraphObject):
        # only set_mask differs in graphs, since nodes and arcs are exactly the same
        # here output mask is used to have a balanced number of targeted nodes in each set mask

        # output masks
        where_output = np.where(dataset.output_mask)[0]
        np.random.shuffle(where_output)
        masks_output = np.array_split(where_output, number_of_batches)

        # set mask with targeted nodes set as False, they will be added during batches construction.
        # if set_mask == output_mask, still True values are added later on.
        set_mask = dataset.set_mask.copy()
        set_mask[dataset.output_mask] = False

        where_set = np.where(set_mask)[0]
        np.random.shuffle(where_set)
        masks_set = np.array_split(where_set, number_of_batches)

        # balanced masks.
        masks = [np.concatenate(i) for i in zip(masks_set, masks_output)]
        for i, _ in enumerate(masks):
            M = masks.copy()

            # append batch masks
            output_dict['training'].append(flatten(M))
            output_dict['validation'].append(M.pop(i - 1) if useVa else np.array([], dtype=int))
            output_dict['test'].append(M.pop(i))

        # qui ritorno un dizionario dove i valori sono gli indici della maschera di set da settare a True

    ### MULTI GRAPH CASE: dataset is a list of graphs or a list of lists of graphs. :param dataset_ for details
    elif isinstance(dataset, list):
        # check type if dataset is a list
        if all(isinstance(i, GraphObject) for i in dataset): dataset = [dataset]
        assert number_of_batches < min([len(i) for i in dataset]), \
            "number of batches must be smaller than len(dataset) or than the number of graphs in the smaller class."
        assert all(isinstance(i, list) for i in dataset) and all(isinstance(i, GraphObject) for i in flatten(dataset))

        # get indices of graphs
        lens = [len(data) for data in dataset]
        lens = [sum(lens[:i]) for i, _ in enumerate(lens)] + [sum(lens)]
        indices = [np.arange(lens[i], lens[i + 1]) for i, _ in enumerate(lens[:-1])]

        # shuffle entire dataset or classes sub-dataset
        for i in indices: np.random.shuffle(i)

        # get dataset batches and flatten lists to obtain a list of lists, then shuffle again to mix classes inside batches
        dataset_batches_indices = list(zip(*[np.array_split(idx, number_of_batches) for idx in indices]))

        # split dataset in training/validation/test set
        for i, _ in enumerate(dataset_batches_indices):
            iTr = dataset_batches_indices.copy()
            iTe = np.concatenate(iTr.pop(i))
            iVa = np.concatenate(iTr.pop(i - 1) if useVa else tuple(np.array([], dtype=int) for _ in iTe))
            iTr = np.concatenate(flatten(iTr))

            # append indices
            output_dict['training'].append(iTr)
            output_dict['validation'].append(iVa)
            output_dict['test'].append(iTe)

        # qui ritorna un dizionario dove i valori sono gli indici del dataset flattenizzato

    else:
        raise TypeError('param <dataset> must be a GraphObject, a list of GraphObjects or a list of lists of Graphobjects')

    output_dict['dataset'] = flatten(dataset)
    return output_dict