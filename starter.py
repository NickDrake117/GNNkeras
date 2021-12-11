import os
import shutil
from typing import Optional, Union

import tensorflow as tf
from numpy import random

from GNN import GNN_utils as utils
from GNN.Models.GNN import GNNnodeBased, GNNedgeBased, GNNgraphBased
from GNN.Sequencers.GraphSequencers import MultiGraphSequencer
from GNN.Models.LGNN import LGNN
from GNN.Models.MLP import MLP, get_inout_dims
from GNN.graph_class import GraphObject

#######################################################################################################################
# SCRIPT OPTIONS - modify the parameters to adapt the execution to the problem under consideration ####################
#######################################################################################################################

# MUTAG option - if True, gnn/lgnn is trained on a real-world dataset MUTAG
# problem is set automatically to graph classification -> addressed_problem='c', problem_based='g'
use_MUTAG: bool = True

# GENERIC GRAPH PARAMETERS. See utils.randomGraph for details
# Node and edge labels are initialized randomly. Target clusters are given by sklearn.
# Each graph has at least <min_nodes_number> nodes and at most <max_nodes_number> nodes
# Possible <aggregation_mode> for matrix ArcNoe belonging to graphs in ['average', 'normalized', 'sum']
# problem_based in ['n', 'a','g'] -> ['c' classification, 'r' regression]
# addressed_problem in ['c', 'r'] -> ['g' graph-based; 'n' node-based; 'a' arc-based;]
problem_based       : str = 'n'
addressed_problem   : str = 'c'
graphs_number       : int = 5
min_nodes_number    : int = 1500
max_nodes_number    : int = 3000
dim_node_label      : int = 5
dim_arc_label       : int = 1
dim_target          : int = 2
density             : float = 0.7
aggregation_mode    : str = 'average'

# LEARNING SETS PARAMETERS
perc_Train      : float = 0.7
perc_Valid      : float = 0.1
batch_size      : int = 32
normalize       : bool = True
seed            : Optional[int] = None
norm_nodes_range: Optional[tuple[Union[int, float], Union[int, float]]] = None  # (-1,1) # other possible value
norm_arcs_range : Optional[tuple[Union[int, float], Union[int, float]]] = None  # (0,1) # other possible value

# NET STATE PARAMETERS
activations_net_state   : str = 'selu'
kernel_init_net_state   : str = 'lecun_normal'
bias_init_net_state     : str = 'lecun_normal'
kernel_reg_net_state    : str = None
bias_reg_net_state      : str = None
dropout_rate_st         : float = 0.1
dropout_pos_st          : Union[list[int], int] = 0
hidden_units_net_state  : Optional[Union[list[int], int]] = None

### NET OUTPUT PARAMETERS
activations_net_output  : str = 'softmax'
kernel_init_net_output  : str = 'glorot_normal'
bias_init_net_output    : str = 'glorot_normal'
kernel_reg_net_output   : str = None
bias_reg_net_output     : str = None
dropout_rate_out        : float = 0.1
dropout_pos_out         : Union[list[int], int] = 0
hidden_units_net_output : Optional[Union[list[int], int]] = None

# GNN PARAMETERS
dim_state       : int = 10
max_iter        : int = 5
state_threshold : float = 0.01

# LGNN PARAMETERS
layers          : int = 5
get_state       : bool = False
get_output      : bool = True
training_mode   : str = 'parallel'

# LOSS / OPTIMIZER PARAMETERS
loss_function   : tf.keras.losses = tf.keras.losses.categorical_crossentropy
optimizer       : tf.keras.optimizers = tf.optimizers.Adam(learning_rate=0.001)

# TRAINING PARAMETERS
# callbacks
path_writer : str = 'writer/'
monitored   : str = 'val_loss'
patience    : int = 10
# training procedure
epochs      : int = 10


#######################################################################################################################
# GPU OPTIONS #########################################################################################################
#######################################################################################################################
# set <use_gpu> parameter in this section in order to use gpu during learning procedure.
# Note that if gpu is not available, use_gpu is set automatically to False.

use_gpu = True
target_gpu = 0

# set target gpu as the only visible device
physical_devices = tf.config.list_physical_devices('GPU')
if use_gpu and len(physical_devices) != 0:
    tf.config.set_visible_devices(physical_devices[int(target_gpu)], device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[int(target_gpu)], True)


#######################################################################################################################
# SCRIPT ##############################################################################################################
#######################################################################################################################

### LOAD DATASET
if use_MUTAG:
    # from MUTAG
    addressed_problem = 'c'
    problem_based = 'g'
    from load_MUTAG import graphs

else:
    # random graphs
    graphs = [utils.randomGraph(nodes_number=int(random.choice(range(min_nodes_number, max_nodes_number))),
                                dim_node_label=dim_node_label,
                                dim_arc_label=dim_arc_label,
                                dim_target=dim_target,
                                density=density,
                                normalize_features=False,
                                aggregation_mode=aggregation_mode,
                                problem_based=problem_based)
              for i in range(graphs_number)]

### PREPROCESSING
# SPLITTING DATASET in Train, Validation and Test set
iTr, iTe, iVa = utils.getindices(len(graphs), perc_Train, perc_Valid, seed=seed)
gTr = [graphs[i] for i in iTr]
gTe = [graphs[i] for i in iTe]
gVa = [graphs[i] for i in iVa]
gGen = gTr[0].copy()

# GRAPHS NORMALIZATION, based on training graphs
if normalize:
    utils.normalize_graphs(gTr, gVa, gTe,
                           based_on='gTr',
                           norm_rangeN=norm_nodes_range,
                           norm_rangeA=norm_arcs_range)

### MODELS
# MLP NETS - STATE
input_net_st, layers_net_st = zip(*[get_inout_dims(net_name='state', dim_node_label=gGen.DIM_NODE_LABEL,
                                                   dim_arc_label=gGen.DIM_ARC_LABEL, dim_target=gGen.DIM_TARGET,
                                                   problem_based=problem_based, dim_state=dim_state,
                                                   hidden_units=hidden_units_net_state,
                                                   layer=i, get_state=get_state, get_output=get_output) for i in range(layers)])
nets_St = [MLP(input_dim=k, layers=j,
               activations=activations_net_state,
               kernel_initializer=kernel_init_net_state,
               bias_initializer=bias_init_net_state,
               kernel_regularizer=kernel_reg_net_state,
               bias_regularizer=bias_reg_net_state,
               dropout_rate=dropout_rate_st,
               dropout_pos=dropout_pos_st,
               name=f'State_{idx}') for idx, (i, j) in enumerate(zip(input_net_st, layers_net_st)) for k in i]

# MLP NETS - OUTPUT
input_net_out, layers_net_out = zip(*[get_inout_dims(net_name='output', dim_node_label=gGen.DIM_NODE_LABEL,
                                                     dim_arc_label=gGen.DIM_ARC_LABEL, dim_target=gGen.DIM_TARGET,
                                                     problem_based=problem_based, dim_state=dim_state,
                                                     hidden_units=hidden_units_net_output,
                                                     layer=i, get_state=get_state, get_output=get_output) for i in range(layers)])
nets_Out = [MLP(input_dim=k, layers=j,
                activations=activations_net_output,
                kernel_initializer=kernel_init_net_output,
                bias_initializer=bias_init_net_output,
                kernel_regularizer=kernel_reg_net_output,
                bias_regularizer=bias_reg_net_output,
                dropout_rate=dropout_rate_out,
                dropout_pos=dropout_pos_out,
                name=f'Out_{idx}') for idx, (i, j) in enumerate(zip(input_net_out, layers_net_out)) for k in i]

# template model
model = {'n': GNNnodeBased, 'a': GNNedgeBased, 'g': GNNgraphBased}[problem_based]

# GNN
gnn = model(nets_St[0], nets_Out[0], dim_state, max_iter, state_threshold).copy()
gnn.compile(optimizer=optimizer, loss=loss_function, average_st_grads=False, metrics=['accuracy', 'mse'], run_eagerly=True)

# LGNN
lgnn = LGNN([model(s, o, dim_state, max_iter, state_threshold) for s, o in zip(nets_St, nets_Out)], get_state, get_output)
lgnn.compile(optimizer=optimizer, loss=loss_function, average_st_grads=True, metrics=['accuracy', 'mse'], run_eagerly=True,
             training_mode=training_mode)

### DATA PROCESSING
# data generator
gTr_Sequencer = MultiGraphSequencer(gTr, problem_based, aggregation_mode, batch_size)
gVa_Sequencer = MultiGraphSequencer(gVa, problem_based, aggregation_mode, batch_size)
gTe_Sequencer = MultiGraphSequencer(gTe, problem_based, aggregation_mode, batch_size)

### TRAINING PROCEDURE
if os.path.exists(path_writer): shutil.rmtree(path_writer)

# callbacks for single gnn
tensorboard_gnn = tf.keras.callbacks.TensorBoard(log_dir=f'{path_writer}single_gnn/', histogram_freq=1)
early_stopping_gnn = tf.keras.callbacks.EarlyStopping(monitor=monitored, restore_best_weights=True, patience=patience)
callbacks_gnn = [tensorboard_gnn, early_stopping_gnn]

# callbacks for lgnn
tensorboard_lgnn = [tf.keras.callbacks.TensorBoard(log_dir=f'{path_writer}gnn{i}/', write_graph=False , histogram_freq=1) for i in range(lgnn.LAYERS)]
early_stopping_lgnn = [tf.keras.callbacks.EarlyStopping(monitor=monitored, restore_best_weights=True, patience=patience) for i in range(lgnn.LAYERS)]
callbacks_lgnn = list(zip(tensorboard_lgnn, early_stopping_lgnn))
if training_mode != 'serial': callbacks_lgnn = callbacks_lgnn[0]

# gnn.fit(gTr_Sequencer, epochs=epochs, validation_data=gVa_Sequencer, callbacks=callbacks_gnn)
# lgnn.fit(gTr_Sequencer, epochs=epochs, validation_data=gVa_Sequencer, callbacks=callbacks_lgnn)
