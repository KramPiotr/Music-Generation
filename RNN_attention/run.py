import os
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from RNN_attention.model import create_network
from utilities.midi_utils import process_midi
from utilities.run_utils import set_run_id
from utilities.utils import get_distinct, retrieve_notes_and_durations, retrieve_network_input_output
from RNN_attention.model_specific_utils import create_store

force_run_id = 'reset' #None #Integer #'reset' #'resetX'
next_run = True
n_if_shortened = None #None #100 #None
epochs = 100 #1

# data params
seq_len = 32
embed_size = 100
rnn_units = 256
use_attention = True

run_id = set_run_id(os.getcwd(), force_run_id, next_run)

# run params
section = 'two_datasets_attention'

section_folder = os.path.join("../run", section)
run_folder = os.path.join(section_folder, run_id)

store_folder = os.path.join(run_folder, 'store')

if not os.path.exists(run_folder):
    os.makedirs(os.path.join(run_folder, 'store'), exist_ok=True)
    os.mkdir(os.path.join(run_folder, 'output'))
    os.mkdir(os.path.join(run_folder, 'weights'))

mode = 'build' # 'build'  # 'load' #

store_model_folder = os.path.join(section_folder, "store")
if mode == 'build':
    create_store("../run/two_datasets_store", store_model_folder, seq_len)

with open(os.path.join(store_model_folder, "distincts"), 'rb') as f:
    [_, n_notes, _, n_durations] = pkl.load(f)

network_input, network_output = retrieve_network_input_output(store_model_folder)

if n_if_shortened:
    for i in range(2):
        network_input[i] = network_input[i][:n_if_shortened]
        network_output[i] = network_output[i][:n_if_shortened]

model, att_model = create_network(n_notes, n_durations, embed_size=embed_size, rnn_units=rnn_units, use_attention=use_attention)
model.summary()
with open(os.path.join(run_folder, "model.txt"),'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

weights_folder = os.path.join(run_folder, 'weights')

checkpoint1 = ModelCheckpoint(
    os.path.join(weights_folder, "weights-improvement-{epoch:02d}-{val_loss:.4f}.h5"),
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)

checkpoint2 = ModelCheckpoint(
    os.path.join(weights_folder, "weights.h5"),
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)

early_stopping = EarlyStopping(
    monitor='val_loss'
    , restore_best_weights=True
    , patience = 3
)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(run_folder, "logs"), histogram_freq=1),

csv_logger = CSVLogger(os.path.join(run_folder, 'log.csv'), append=True, separator=';')

callbacks_list = [
    checkpoint1
    , checkpoint2
    , early_stopping
    , tensorboard
    , csv_logger
 ]
model.save_weights(os.path.join(weights_folder, "weights.h5"))
history_callback = model.fit(network_input, network_output
          , epochs=epochs, batch_size=32
          , validation_split = 0.2
          , callbacks=callbacks_list
          , shuffle=True
         )
