from typing import Union, Optional

from numpy import array, arange, sum
from tensorflow.keras.layers import Dense, Dropout, AlphaDropout, BatchNormalization
from tensorflow.keras.models import Sequential


# ---------------------------------------------------------------------------------------------------------------------
def MLP(input_dim: int, layers: list[int], activations, kernel_initializer, bias_initializer,
        kernel_regularizer=None, bias_regularizer=None, dropout_rate: Union[list[float], float, None] = None,
        dropout_pos: Optional[Union[list[int], int]] = None, alphadropout: bool = False, batch_normalization: bool = True, *, name: str = None):
    """ Quick building function for MLP model. All lists must have the same length

    :param input_dim: (int) specify the input dimension for the model
    :param layers: (int or list of int) specify the number of units in every layers
    :param activations: (functions or list of functions)
    :param kernel_initializer: (initializers or list of initializers) for weights initialization (NOT biases)
    :param bias_initializer: (initializers or list of initializers) for biases initialization (NOT weights)
    :param kernel_regularizer: (regularizer or list of regularizers) for weight regularization (NOT biases)
    :param bias_regularizer: (regularizer or list of regularizers) for biases regularization (NOT weights)
    :param dropout_rate: (float) s.t. 0 <= dropout_percs <= 1 for dropout rate
    :param dropout_pos: int or list of int describing dropout layers position
    :param alphadropout: (bool) for dropout type, if any
    :param batch_normalization: (bool) add a BatchNormalization layer after the last dense layer
    :return: Sequential (MLP) model
    """

    # build lists
    if type(activations) != list: activations = [activations for _ in layers]
    if type(kernel_initializer) != list: kernel_initializer = [kernel_initializer for _ in layers]
    if type(bias_initializer) != list: bias_initializer = [bias_initializer for _ in layers]
    if type(kernel_regularizer) != list: kernel_regularizer = [kernel_regularizer for _ in layers]
    if type(bias_regularizer) != list: bias_regularizer = [bias_regularizer for _ in layers]
    if type(dropout_pos) == int:  dropout_pos = [dropout_pos]
    if type(dropout_rate) == float: dropout_rate = [dropout_rate for _ in dropout_pos]
    if dropout_rate == None or dropout_pos == None: dropout_rate, dropout_pos = list(), list()

    # check lengths
    if len(set(map(len, [activations, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, layers]))) > 1:
        raise ValueError('Dense parameters must have the same length to be correctly processed')
    if len(dropout_rate) != len(dropout_pos):
        raise ValueError('Dropout parameters must have the same length to be correctly processed')

    # Dense layers
    if name is None: dense_names = [None for _ in layers]
    else: dense_names = [f'{name}_dense_{i}' for i, _ in enumerate(layers)]
    keys = ['units', 'activation', 'kernel_initializer', 'bias_initializer', 'kernel_regularizer', 'bias_regularizer', 'name']
    vals = zip(layers, activations, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, dense_names)

    params_layers = [dict(zip(keys, i)) for i in vals]
    keras_layers = [Dense for _ in params_layers]

    # Dropout layers
    if dropout_rate and dropout_pos:
        dropout = AlphaDropout if alphadropout else Dropout
        dropout_pos = array(dropout_pos, dtype=int) + arange(len(dropout_pos))
        for i, (rate, pos) in enumerate(zip(dropout_rate, dropout_pos)):
            keras_layers.insert(pos, dropout)
            params_layers.insert(pos, {'rate': rate, 'name': f'{name}_dropout_{i}'})

    # Batch normalization layer
    if batch_normalization:
        batch_normalization_name = f'{name}_batch_normalization' if name is not None else None
        params_layers.insert(0, {'name': batch_normalization_name})
        keras_layers.insert(0, BatchNormalization)

    # set input shape for first layer
    params_layers[0]['input_shape'] = input_dim

    # return MLP model
    mlp_layers = [layer(**params) for layer, params in zip(keras_layers, params_layers)]

    return Sequential(mlp_layers, name=name)


# ---------------------------------------------------------------------------------------------------------------------
def get_inout_dims(net_name: str, dim_node_label: int, dim_arc_label: int, dim_target: int, problem_based: str, dim_state: int,
                   hidden_units: Union[None, int, list[int]],
                   *, layer: int = 0, get_state: bool = False, get_output: bool = False) -> tuple[list[tuple], list[int]]:
    """ Calculate input and output dimension for the MLP of state and output

    :param net_name: (str) in ['state','output']
    :param dim_node_label: (int) dimension of node label
    :param dim_arc_label: (int) dimension of arc label
    :param dim_target: (int) dimension of target
    :param problem_based: (str) s.t. len(problem_based) in [1,2] -> [{'a','n','g'} | {'1','2'}]
    :param dim_state: (int)>=0 for state dimension paramenter of the gnn
    :param hidden_units: (int or list of int) for specifying units on hidden layers
    :param layer: (int) LGNN USE: get the dims at gnn of the layer <layer>, from graph dims on layer 0. Default is 0, since GNN==LGNN in this case
    :param get_state: (bool) LGNN USE: set accordingly to LGNN behaviour, if gnns get state, output or both from previous layer
    :param get_output: (bool) LGNN USE: set accordingly to LGNN behaviour, if gnns get state, output or both from previous layer
    :return: (tuple) (input_shape, layers) s.t. input_shape (int) is the input shape for mlp, layers (list of ints) defines hidden+output layers
    """
    assert layer >= 0
    assert problem_based in ['a', 'n', 'g']
    assert dim_state >= 0
    assert isinstance(hidden_units, (int, type(None))) or (isinstance(hidden_units, list) and all(isinstance(x, int) for x in hidden_units))

    #NL, AL, T = dim_node_label, dim_arc_label, dim_target
    NL, AL, T = array(dim_node_label, ndmin=1), dim_arc_label, dim_target
    DS, GS, GO = dim_state, get_state, get_output

    # if LGNN, get MLPs layers for gnn in layer 2+
    if layer > 0:
        if DS != 0:
            NL = NL + DS * GS + T * (problem_based != 'a') * GO
            AL = AL + T * (problem_based == 'a') * GO
        else:
            NL = NL + layer * NL * GS + ((layer - 1) * GS + 1) * T * (problem_based != 'a') * GO
            AL = AL + T * (problem_based == 'a') * GO

    # MLP state
    if net_name == 'state':
        NLgen = sum(NL)
        input_shape = list(NL + NLgen + AL + 2 * DS)
        output_shape = DS if DS else NL

    # MLP output
    elif net_name == 'output':
        if len(NL)>1: NL = array([0]) ### è QUI IL PROBLEMA, un NL di sotto rimane != da 0
        input_shape =  list((problem_based == 'a') * (NL + AL + DS) + NL + DS)
        output_shape = T

    # possible values for net_name in ['state','output'], otherwise raise error
    else: raise ValueError(':param net_name: not in [\'state\', \'output\']')

    # input shape: generale case
    # if the problem is NON-composite, it is a list with a single tuple of len==1, a list of tuples of len==1 each otherwise
    input_shape = [(i,) for i in input_shape]

    # hidden part
    if not hidden_units: hidden_units = list()
    if isinstance(hidden_units, int): hidden_units = [hidden_units]
    layers = hidden_units + [output_shape]

    return input_shape, layers