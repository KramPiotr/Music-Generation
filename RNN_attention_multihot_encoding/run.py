import os
import pickle as pkl
import glob
from RNN_attention_multihot_encoding.utils import open_midi, extract_notes, prepare_sequences, create_lookups, get_distinct
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model
from RNN_attention_multihot_encoding.model import create_network

next_run = True

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



# run params
section = 'MIREX_multihot'

run_folder = os.path.join("../run", section)
run_folder = os.path.join(run_folder, run_id)

store_folder = os.path.join(run_folder, 'store')

if not os.path.exists(run_folder):
    os.makedirs(os.path.join(run_folder, 'store'), exist_ok=True)
    os.mkdir(os.path.join(run_folder, 'output'))
    os.mkdir(os.path.join(run_folder, 'weights'))

mode = 'load' # 'build'  # 'load' #

# data params
seq_len = 32
embed_size = 3
rnn_units = 256
use_attention = True
n_notes = 12

if mode == 'build':

    music_list = glob.glob("../datasets/MIREX_dataset/MIDIs/*.mid")
    print(len(music_list), 'files in total')

    notes = []
    durations = []

    for i, file in enumerate(music_list):
        print(i + 1, "Parsing %s" % file)
        original_score = open_midi(file)
        n, d = extract_notes(original_score, seq_len)
        notes.extend(n)
        durations.extend(d)

    with open(os.path.join(store_folder, 'notes'), 'wb') as f:
        pkl.dump(notes, f)
    with open(os.path.join(store_folder, 'durations'), 'wb') as f:
        pkl.dump(durations, f)
    # get the distinct sets of notes and durations
    duration_names, n_durations = get_distinct(durations)
    distincts = [duration_names, n_durations]

    with open(os.path.join(store_folder, 'distincts'), 'wb') as f:
        pkl.dump(distincts, f)

    # make the lookup dictionaries for notes and dictionaries and save
    duration_to_int, int_to_duration = create_lookups(duration_names)
    lookups = [duration_to_int, int_to_duration]

    with open(os.path.join(store_folder, 'lookups'), 'wb') as f:
        pkl.dump(lookups, f)

    network_input, network_output = prepare_sequences(notes, durations, lookups, distincts, seq_len)

    with open(os.path.join(store_folder, 'network_input'), 'wb') as f:
        pkl.dump(network_input, f)

    with open(os.path.join(store_folder, 'network_output'), 'wb') as f:
        pkl.dump(network_output, f)
else:
    # with open(os.path.join(store_folder, 'notes'), 'rb') as f:
    #     notes = pickle.load(f)
    retrieve_folder = "../run/MIREX_multihot/00/store"

    with open(os.path.join(retrieve_folder, 'durations'), 'rb') as f:
        durations = pkl.load(f)

    with open(os.path.join(retrieve_folder, 'distincts'), 'rb') as f:
        duration_names, n_durations = pkl.load(f)

    with open(os.path.join(retrieve_folder, 'network_input'), 'rb') as f:
        network_input = pkl.load(f)

    with open(os.path.join(retrieve_folder, 'network_output'), 'rb') as f:
        network_output = pkl.load(f)



model, att_model = create_network(n_notes, n_durations, seq_len=seq_len, embed_size=embed_size, rnn_units=rnn_units, use_attention=use_attention)
model.summary()

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


callbacks_list = [
    checkpoint1
    , checkpoint2
    , early_stopping
    , tensorboard
 ]

model.save_weights(os.path.join(weights_folder, "weights.h5"))
model.fit(network_input, network_output
          , epochs=200, batch_size=32
          , validation_split = 0.2
          , callbacks=callbacks_list
          , shuffle=True
         )

#TODO
#train on only a few samples and test
#analyze the notes and durations data
#build simple classifiers
#understand what happens with the attention
#hype yourself - its machine learning!
#train the same stuff on Google Colab and compare speed
#standarize your code (format + pickle as pkl)