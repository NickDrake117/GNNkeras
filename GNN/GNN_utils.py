import os.path
from typing import Union, Optional

import numpy as np

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
        assert number_of_batches <= min([len(i) for i in dataset]), \
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

# ---------------------------------------------------------------------------------------------------------------------
def LKO(model, lko_dict: dict, focus: str, aggregation_mode: str, sequencer, sequencer_kwargs, epochs: int, scalers_dict: dict, LGNN_case: bool,
        compile_kwargs: dict, save_model: bool = True, verbose=2, output_folder='LKO'):
    from tensorflow.keras import callbacks
    from pandas import DataFrame

    if output_folder[-1] != '/': output_folder += '/'
    if os.path.exists(output_folder) and len(os.listdir(output_folder))>1:
        raise ValueError(':param output_folder: already exists. Please change LKO output location folder and re-run.')

    dataset = lko_dict.pop('dataset')
    res = list()
    for idx, elems in enumerate(zip(*lko_dict.values())):
        print(f"\n\nBATCH No. {idx + 1}/{len(lko_dict['training'])}")

        ### get current test/training/validation sets and normalize data if needed 
        gtr, gva, gte = [[dataset[i].copy() for i in j] for j in elems]
        if scalers_dict is not None:
            G = GraphObject.merge(gtr, focus, aggregation_mode)
            scalers = G.normalize(scalers_dict, True)
            for g in gtr + gva + gte: g.normalize_from(scalers=scalers)

        _ = sequencer_kwargs.pop('shuffle', False)
        training_sequencer = sequencer(gtr, **sequencer_kwargs, shuffle=False)
        validation_sequencer = sequencer(gva, **sequencer_kwargs, shuffle=False)
        test_sequencer = sequencer(gte, **sequencer_kwargs, shuffle=False)

        ### defines callbacks
        if LGNN_case:
            # callbacks for lgnn
            path = f"{output_folder}LGNN{idx}/"
            tb = [callbacks.TensorBoard(log_dir=f'{path}gnn{i}/', histogram_freq=1) for i in range(model.LAYERS)]
            es = [callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True) for i in range(model.LAYERS)]

            if model.training_mode != 'serial': tb, es = [tb[0]], [es[0]]

        else:
            # callbacks for gnn
            path = f"{output_folder}GNN{idx}/"
            tb = [callbacks.TensorBoard(log_dir=f'{path}', histogram_freq=1)]
            es = [callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)]

        # final callbacks
        cb = list(zip(tb, es))

        ### re-initialize model + training model with training batches data
        model_tmp = model.copy()
        model_tmp.compile(loss=model.compiled_loss._losses, metrics=model.compiled_metrics._metrics, **compile_kwargs)
        model_tmp.fit(training_sequencer, epochs=epochs, validation_data=validation_sequencer, callbacks=cb, verbose=verbose)

        # save model for batch
        if save_model: model.save(f'{path}model')

        ### test model on current test batch
        model_prediction = model.predict(test_sequencer, verbose=verbose)
        targets = test_sequencer.targets #np.concatenate([g.targets for g in test_sequencer.data], axis=0)

        # res dataframe
        res_model_tmp = DataFrame(np.vstack([targets.flatten(), model_prediction.flatten()]).transpose(), columns=['Targs', 'Out'])
        res.append(model_tmp.evaluate(test_sequencer, return_dict=True, verbose=verbose))

        # add columns of (real, not normalized before) targets and de-normalized output
        if scalers_dict is not None and 'targets' in scalers_dict:
            targets = np.concatenate([dataset[k].targets[g.output_mask].copy() for k,g in zip(elems[-1], test_sequencer._data)], axis=0)
            res_model_tmp = DataFrame(
                np.concatenate([res_model_tmp.values,
                                np.vstack([targets.flatten(),
                                           scalers['targets'].inverse_transform(model_prediction).flatten()]).transpose()], axis=1),
                columns=['Targs', 'Out', 'Real Targs', 'Real Out'])

        ### print batch results
        print('\n\n')
        res_model_tmp.to_csv(f"{path}test_res.csv", index=False)
        print(DataFrame.from_dict(res[:idx + 1]))

    return DataFrame.from_dict(res)