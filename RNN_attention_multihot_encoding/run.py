import os
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from RNN_attention_multihot_encoding.model import create_network
from utilities.midi_utils import process_midi

force_run_id = '0' #None #Integer #'reset' #'resetX'
next_run = True
n_if_shortened = None #100 #None
epochs = 100 #1

# data params
seq_len = 32
embed_size = 3
rnn_units = 256
use_attention = True
n_notes = 12


if not force_run_id:
    run_id_file = "run_id"
    with open(run_id_file, 'rb') as f:
        run_id = pkl.load(f)
    if next_run:
        run_id += 1
    with open(run_id_file, 'wb') as f:
        pkl.dump(run_id, f)
    if run_id < 10:
        run_id = '0' + str(run_id)
    else:
        run_id = str(run_id)
else:
    run_id = force_run_id


# run params
section = 'two_datasets_multihot'

run_folder = os.path.join("../run", section)
run_folder = os.path.join(run_folder, run_id)

store_folder = os.path.join(run_folder, 'store')

if not os.path.exists(run_folder):
    os.makedirs(os.path.join(run_folder, 'store'), exist_ok=True)
    os.mkdir(os.path.join(run_folder, 'output'))
    os.mkdir(os.path.join(run_folder, 'weights'))

mode = 'load' # 'build'  # 'load' #

if mode == 'build':
    process_midi("../datasets/MIREX_dataset/MIDIs", seq_len, store_folder)
else:
    # with open(os.path.join(store_folder, 'notes'), 'rb') as f:
    #     notes = pickle.load(f)
    retrieve_folder = "../run/two_datasets_store"

    with open(os.path.join(retrieve_folder, 'durations'), 'rb') as f:
        durations = pkl.load(f)

    with open(os.path.join(retrieve_folder, 'distincts'), 'rb') as f:
        duration_names, n_durations = pkl.load(f)

    with open(os.path.join(retrieve_folder, 'network_input'), 'rb') as f:
        network_input = pkl.load(f)

    with open(os.path.join(retrieve_folder, 'network_output'), 'rb') as f:
        network_output = pkl.load(f)

if n_if_shortened:
    for i in range(2):
        network_input[i] = network_input[i][:n_if_shortened]
        network_output[i] = network_output[i][:n_if_shortened]

model, att_model = create_network(n_notes, n_durations, seq_len=seq_len, embed_size=embed_size, rnn_units=rnn_units, use_attention=use_attention)
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
    checkpoint1,
    checkpoint2,
    early_stopping,
    tensorboard,
    csv_logger,
 ]
model.save_weights(os.path.join(weights_folder, "weights.h5"))
history_callback = model.fit(network_input, network_output
          , epochs=epochs, batch_size=32
          , validation_split = 0.2
          , callbacks=callbacks_list
          , shuffle=True
         )

# loss_history = history_callback.history["val_pitch_loss"]
# numpy_loss_history = np.array(loss_history)
# np.savetxt("val_pitch_loss_history.txt", numpy_loss_history, delimiter=",")
#TODO
#train on only a few samples and test
#analyze the notes and durations data
#build simple classifiers
#understand what happens with the attention
#hype yourself - its machine learning!
#train the same stuff on Google Colab and compare speed
#standarize your code (format + pickle as pkl)