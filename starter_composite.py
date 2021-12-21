# codinf=utf-8
import tensorflow as tf

from numpy import random
from GNN.Models.MLP import MLP, get_inout_dims
from GNN.Models.CompositeGNN import CompositeGNNgraphBased
from GNN.Models.CompositeLGNN import CompositeLGNN
from GNN.Sequencers.GraphSequencers import CompositeMultiGraphSequencer


#######################################################################################################################
# SCRIPT OPTIONS - modify the parameters to adapt the execution to the problem under consideration ####################
#######################################################################################################################

### GRAPHS OPTIONS
aggregation_mode = 'average'
# c: Classification
addressed_problem = 'c'
# g: graph-based
problem_based = 'g'

# NET STATE PARAMETERS
activations_net_state   : str = 'selu'
kernel_init_net_state   : str = 'lecun_normal'
bias_init_net_state     : str = 'lecun_normal'
### NET OUTPUT PARAMETERS
activations_net_output  : str = 'softmax'
kernel_init_net_output  : str = 'glorot_normal'
bias_init_net_output    : str = 'glorot_normal'

# GNN PARAMETERS
dim_state       : int = 10
max_iter        : int = 5
state_threshold : float = 0.01

# LGNN PARAMETERS
layers          : int = 5
get_state       : bool = True
get_output      : bool = True
training_mode   : str = 'parallel'

# LEARNING PARAMETERS
epochs          : int = 5
batch_size      : int = 500
loss_function   : tf.keras.losses = tf.keras.losses.categorical_crossentropy
optimizer       : tf.keras.optimizers = tf.optimizers.Adam(learning_rate=0.01)


#######################################################################################################################
# SCRIPT ##############################################################################################################
#######################################################################################################################

### LOAD DATASET from MUTAG
# problem is set automatically to graph-based one -> problem_based='g'
# then aggregation_mode is set for each graph, since they are initialized with aggregation_mode = 'average',
# but one can choose between 'average', 'sum', 'normalized'.
from load_MUTAG import composite_graphs as graphs
for g in graphs: g.setAggregation(aggregation_mode)


### PREPROCESSING
# SPLITTING DATASET in Train, Validation and Test set, no graph normalization is applied
random.shuffle(graphs)
gTr = graphs[:-1500]
gTe = graphs[-1500:-750]
gVa = graphs[-750:]
gGen = gTr[0].copy()

### MODELS
# NETS - STATE
input_net_st, layers_net_st = zip(*[get_inout_dims(net_name='state', dim_node_label=gGen.DIM_NODE_LABEL,
                                                   dim_arc_label=gGen.DIM_ARC_LABEL, dim_target=gGen.DIM_TARGET,
                                                   problem_based=problem_based, dim_state=dim_state,
                                                   layer=i, get_state=get_state, get_output=get_output) for i in range(layers)])
nets_St = [[MLP(input_dim=s, layers=j,
                activations=activations_net_state,
                kernel_initializer=kernel_init_net_state,
                bias_initializer=bias_init_net_state,
                name=f'State_{idx}') for s in i] for idx, (i, j) in enumerate(zip(input_net_st, layers_net_st))]

# NETS - OUTPUT
nets_Out = MLP(input_dim=(dim_state,), layers=[2],
                activations=activations_net_output,
                kernel_initializer=kernel_init_net_output,
                bias_initializer=bias_init_net_output,
                name='Out_1')

# GNN
gnn = CompositeGNNgraphBased(nets_St[0], nets_Out, dim_state, max_iter, state_threshold).copy()
gnn.compile(optimizer=optimizer, loss=loss_function, average_st_grads=False, metrics=['accuracy', 'mse'], run_eagerly=True)

# LGNN
lgnn = CompositeLGNN([CompositeGNNgraphBased(s, nets_Out, dim_state, max_iter, state_threshold) for s in nets_St], get_state, get_output)
lgnn.compile(optimizer=optimizer, loss=loss_function, average_st_grads=True, metrics=['accuracy', 'mse'], run_eagerly=True,
             training_mode=training_mode)

### DATA PROCESSING
gTr_Sequencer = CompositeMultiGraphSequencer(gTr, problem_based, aggregation_mode, batch_size)
gVa_Sequencer = CompositeMultiGraphSequencer(gVa, problem_based, aggregation_mode, batch_size)
gTe_Sequencer = CompositeMultiGraphSequencer(gTe, problem_based, aggregation_mode, batch_size)

### LEARNING PROCEDURE
# gnn.fit(gTr_Sequencer, epochs=epochs, validation_data=gVa_Sequencer)
# lgnn.fit(gTr_Sequencer, epochs=epochs, validation_data=gVa_Sequencer)
