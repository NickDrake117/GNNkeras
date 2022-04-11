# coding=utf-8

import shutil

import numpy as np

from GNN.Models.MLP import MLP, get_inout_dims
from options import *


#######################################################################################################################
# FUNCTIONS ###########################################################################################################
#######################################################################################################################
def print_setup(dir):
    setup = {i: eval(i) for i in dir if '__' not in i and 'module' not in str(eval(i)) and 'typing.' not in str(eval(i))}
    lm = max([len(i) for i in setup])
    print("\n\n\n=========== SETUP ===========")
    for i in setup: print(f"{i:{lm}}\t\t", setup[i])
    print("========= END SETUP =========")


def save_setup(dir, filepath):
    from time import asctime
    setup = {i: eval(i) for i in dir if '__' not in i and 'module' not in str(eval(i)) and 'typing.' not in str(eval(i))}
    lm = max([len(i) for i in setup])
    with open(f"{filepath}{'.txt' if '.txt' not in filepath else ''}", 'w') as scrivi:
        scrivi.write(asctime())
        scrivi.write("\n\n=========== SETUP ===========\n")
        for i in setup: scrivi.write(f"{i:{lm}}\t\t{setup[i]}\n")
        scrivi.write("========= END SETUP =========")


#######################################################################################################################
# SCRIPT ##############################################################################################################
#######################################################################################################################

####### PRINT SETUP OPTION
PRINT_SETUP: bool = True
if PRINT_SETUP: print_setup(setup)

####### SET DTYPE
tf.keras.backend.set_floatx(graph_dtype)

####### LOAD DATASET
from load_MUTAG import graphs
np.random.shuffle(graphs)
graphs = graphs[:100]

gTr, gVa, gTe = np.array_split(graphs, 3)
gTr = gTr.tolist()
gVa = gVa.tolist()
gTe = gTe.tolist()
gGen = gTr[0]

####### MODELS
### MLP NETS - STATE
net_state_input, net_state_layers = zip(*[get_inout_dims(net_name='state',
                                                         dim_node_features=gGen.DIM_NODE_FEATURES,
                                                         dim_arc_label=gGen.DIM_ARC_FEATURES,
                                                         dim_target=gGen.DIM_TARGET,
                                                         focus=graph_focus,
                                                         dim_state=gnn_dim_state,
                                                         hidden_units=net_state_hidden_units,
                                                         layer=i,
                                                         get_state=lgnn_get_state,
                                                         get_output=lgnn_get_output)
                                          for i in range(lgnn_layers)])

nets_state = [MLP(input_dim=k, layers=j, activations=net_state_activations,
                  kernel_initializer=net_state_kernel_init, bias_initializer=net_state_bias_init,
                  kernel_regularizer=net_state_kernel_reg, bias_regularizer=net_state_bias_reg,
                  dropout_rate=net_state_dropout_rate, dropout_pos=net_state_dropout_pos, alphadropout=net_state_alphadropout,
                  batch_normalization=net_state_batch_norm,
                  name=f'State_{idx}')
              for idx, (i, j) in enumerate(zip(net_state_input, net_state_layers)) for k in i]

### MLP NETS - OUTPUT
net_output_input, net_output_layers = zip(*[get_inout_dims(net_name='output',
                                                           dim_node_features=gGen.DIM_NODE_FEATURES,
                                                           dim_arc_label=gGen.DIM_ARC_FEATURES,
                                                           dim_target=gGen.DIM_TARGET,
                                                           focus=graph_focus,
                                                           dim_state=gnn_dim_state,
                                                           hidden_units=net_output_hidden_units,
                                                           layer=i,
                                                           get_state=lgnn_get_state,
                                                           get_output=lgnn_get_output)
                                            for i in range(lgnn_layers)])

nets_output = [MLP(input_dim=k, layers=j, activations=net_output_activations,
                   kernel_initializer=net_output_kernel_init, bias_initializer=net_output_bias_init,
                   kernel_regularizer=net_output_kernel_reg, bias_regularizer=net_output_bias_reg,
                   dropout_rate=net_output_dropout_rate, dropout_pos=net_output_dropout_pos,
                   alphadropout=net_output_alphadropout,
                   batch_normalization=net_output_batch_norm,
                   name=f'Out_{idx}')
               for idx, (i, j) in enumerate(zip(net_output_input, net_output_layers)) for k in i]

# PATH WRITER - TENSORBOARD
if os.path.exists(learning_path_writer): shutil.rmtree(learning_path_writer)

### GNN
gnn = gnn_model(nets_state[0], nets_output[0], gnn_dim_state, gnn_max_iter, gnn_state_threshold).copy()
gnn.compile(optimizer=learning_optimizer, loss=learning_loss_function, metrics=learning_metrics,
            average_st_grads=learning_avg_state_grads, run_eagerly=learning_run_eagerly)
# callbacks for single gnn
es_gnn = tf.keras.callbacks.EarlyStopping(monitor=learning_monitored, mode='auto', verbose=1, patience=learning_patience)
tensorboard_gnn = tf.keras.callbacks.TensorBoard(log_dir=f'{learning_path_writer}single_gnn/', histogram_freq=1)
callbacks_gnn = [tensorboard_gnn, es_gnn]

### LGNN
lgnn = lgnn_model([gnn_model(s, o, gnn_dim_state, gnn_max_iter, gnn_state_threshold) for s, o in zip(nets_state, nets_output)],
                  lgnn_get_state, lgnn_get_output)
lgnn.compile(optimizer=learning_optimizer, loss=learning_loss_function, metrics=learning_metrics,
             training_mode=lgnn_training_mode, average_st_grads=learning_avg_state_grads, run_eagerly=learning_run_eagerly)

# callbacks for lgnn
if lgnn_training_mode != 'serial': lgnn_layers = 1
tensorboard_lgnn = [tf.keras.callbacks.TensorBoard(log_dir=f'{learning_path_writer}gnn{i}/', histogram_freq=1) for i in range(lgnn_layers)]
es_lgnn = [tf.keras.callbacks.EarlyStopping(monitor=learning_monitored, mode='auto', verbose=1, patience=learning_patience) for i in
           range(lgnn_layers)]
callbacks_lgnn = list(zip(tensorboard_lgnn, es_lgnn))

### DATA PROCESSING
training_sequencer = sequencer_model(gTr, **sequencer_kwargs)
validation_sequencer = sequencer_model(gVa, **sequencer_kwargs)
test_sequencer = sequencer_model(gTe, **sequencer_kwargs)

### LEARNING PROCEDURE
# gnn.fit(training_sequencer, epochs=epochs, validation_data=validation_sequencer, callbacks=callbacks_gnn)
# lgnn.fit(training_sequencer, epochs=epochs, validation_data=validation_sequencer)
