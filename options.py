# codinf=utf-8
import os

import tensorflow as tf
import tensorflow.keras.metrics as kmt
#import tensorflow_addons.metrics as mt

from typing import Union, Optional
from sklearn import preprocessing
from sklearn import metrics as skmt

from GNN.Models import GNN, CompositeGNN, LGNN, CompositeLGNN
from GNN.Sequencers import GraphSequencers, TransductiveGraphSequencers


#######################################################################################################################
# SCRIPT OPTIONS - modify the parameters to adapt the execution to the problem under consideration ####################
#######################################################################################################################

### GPU + general OPTION
os.environ["CUDA_VISIBLE_DEVICES"]: str  = "0"
composite_case                    : bool = True

### GRAPHS OPTIONS
graph_dataset_path      : str   = "grafi_sym"
graph_aggregation_mode  : str   = 'average'
graph_addressed_problem : str   = 'c'
graph_focus             : str   = 'g'
graph_dtype             : str   = 'float32'
graph_scalers           : dict  = {'nodes'  : {'class': preprocessing.MinMaxScaler, 'kwargs': {'feature_range': (0, 1)}},
                                   'targets': {'class': preprocessing.MinMaxScaler, 'kwargs': {'feature_range': (0, 1)}}}

# NET STATE PARAMETERS
net_state_activations   : str   = 'linear'
net_state_kernel_init   : str   = 'lecun_uniform'
net_state_bias_init     : str   = 'lecun_uniform'
net_state_kernel_reg    : str   = None
net_state_bias_reg      : str   = None
net_state_batch_norm    : bool  = True
net_state_dropout_rate  : float = 0.1
net_state_dropout_pos   : Union[list[int], int] = 0
net_state_alphadropout  : bool  = False
net_state_hidden_units  : Optional[Union[list[int], int]] = None#[256]

### NET OUTPUT PARAMETERS
net_output_activations  : str   = 'softmax'
net_output_kernel_init  : str   = 'glorot_uniform'
net_output_bias_init    : str   = 'glorot_uniform'
net_output_kernel_reg   : str   = None
net_output_bias_reg     : str   = None
net_output_batch_norm   : bool  = True
net_output_dropout_rate : float = 0.1
net_output_dropout_pos  : Union[list[int], int] = 0
net_output_alphadropout : bool  = False
net_output_hidden_units : Optional[Union[list[int], int]] = None#[100, 100, 100]

# GNN PARAMETERS
gnn_dim_state           : int   = 10
gnn_max_iter            : int   = 2
gnn_state_threshold     : float = 0.01

# LGNN PARAMETERS
lgnn_layers             : int   = 3
lgnn_get_state          : bool  = 1
lgnn_get_output         : bool  = 1
lgnn_training_mode      : str   = 'serial'

# SEQUENCER PARAMETER
if composite_case:  sequencer_model     : GraphSequencers = GraphSequencers.CompositeMultiGraphSequencer
else:               sequencer_model     : GraphSequencers = GraphSequencers.MultiGraphSequencer
sequencer_kwargs    : dict = {'focus': graph_focus, 'aggregation_mode':graph_aggregation_mode, 'batch_size':200, 'shuffle':False}

# LEARNING PARAMETERS
learning_run_eagerly    : bool  = False
learning_epochs         : int   = 5
learning_loss_function  : tf.keras.losses       = 'categorical_crossentropy'
learning_optimizer      : tf.keras.optimizers   = tf.keras.optimizers.Adam(learning_rate=0.01)
learning_path_writer    : str   = 'writer/'
learning_monitored      : str   = 'val_loss'
learning_patience       : int   = 50
learning_avg_state_grads: bool  = False
learning_metrics        : list  = None#['mse', 'msle', 'mae', 'cosine_similarity', mt.r_square.RSquare(y_shape=(4,))]

#### SETUP - NO NEED TO SET PARAMETERS FROM HERE
lgnn_model = CompositeLGNN.CompositeLGNN                if composite_case   else LGNN.LGNN
gnn_model  = {'n': CompositeGNN.CompositeGNNnodeBased   if composite_case   else GNN.GNNnodeBased,
              'a': CompositeGNN.CompositeGNNarcBased    if composite_case   else GNN.GNNarcBased,
              'g': CompositeGNN.CompositeGNNgraphBased  if composite_case   else GNN.GNNgraphBased}[graph_focus]
setup = dir()