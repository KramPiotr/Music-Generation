# Author: Piotr Kram
import numpy as np
from tensorflow import keras
from keras.utils import np_utils
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.utils import plot_model
import os
import pickle as pkl
# print(os.getcwd())
import sys
import shutil

sys.path.append(os.path.dirname(os.getcwd()))
from utilities.notes_utils import multi_hot_encoding_12_tones
import ntpath
import sys
import re
from glob import glob
from pathlib import Path, PureWindowsPath
from functools import partial
import RNN_attention.model  # helps with retrieving the model
import RNN_attention_multihot_encoding.model  # helps with retrieving the model


def get_distinct(elements):
    # Get all pitch names
    element_names = sorted(set(elements))  # a lot of 0s for duration...
    n_elements = len(element_names)
    return (element_names, n_elements)


def create_lookups(element_names):
    # create dictionary to map notes and durations to integers
    element_to_int = dict((element, number) for number, element in enumerate(element_names))
    int_to_element = dict((number, element) for number, element in enumerate(element_names))

    return (element_to_int, int_to_element)


def prepare_sequences(notes, durations, lookups, distincts, seq_len=32):
    """ Prepare the sequences used to train the Neural Network """

    duration_to_int, int_to_duration = lookups
    duration_names, n_durations = distincts
    encoded_notes = multi_hot_encoding_12_tones(notes)

    notes_network_input = []
    notes_network_output = []
    durations_network_input = []
    durations_network_output = []

    # create input sequences and the corresponding outputs
    for i in range(len(notes) - seq_len):
        notes_sequence_in = encoded_notes[i:i + seq_len]
        notes_sequence_out = encoded_notes[i + seq_len]
        notes_network_input.append(notes_sequence_in)
        notes_network_output.append(notes_sequence_out)

        durations_sequence_in = durations[i:i + seq_len]
        durations_sequence_out = durations[i + seq_len]
        durations_network_input.append([duration_to_int[char] for char in durations_sequence_in])
        durations_network_output.append(duration_to_int[durations_sequence_out])

    # reshape the input into a format compatible with LSTM layers
    notes_network_input = np.array(notes_network_input)
    durations_network_input = np.array(durations_network_input)
    network_input = [notes_network_input, durations_network_input]

    notes_network_output = np.array(notes_network_output)
    durations_network_output = np_utils.to_categorical(durations_network_output, num_classes=n_durations)
    network_output = [notes_network_output, durations_network_output]

    return network_input, network_output


def find_interpolated(val, arr, out_arr):
    idx = np.searchsorted(arr, val)
    ld = val - arr[idx - 1] if idx > 0 else val
    rd = arr[idx] - val if idx < len(arr) else 1 - val
    lo = out_arr[idx - 1] if idx > 0 else 0
    ro = out_arr[idx] if idx < len(arr) else 1
    return np.interp(0, (-ld, rd), (lo, ro))


# def multihot_sample(preds, temperature, firing_loc):
#     firing_percentiles = np.array(retrieve(firing_loc, "firing_percentiles"))
#     pcntl = np.array(retrieve(firing_loc, "percentile_names"))
#     firing_fractions = np.array(retrieve(firing_loc, "firing_fractions"))
#     for i, p in enumerate(preds):
#         frac_off = find_interpolated(p, firing_percentiles[i][0], pcntl)
#         pctnl_on = find_interpolated(p, firing_percentiles[i][1], pcntl)
#
#     fraction = np.random.uniform(0, 1, len(preds)) >= firing_fractions
#     is_on = preds >= firing_tresholds
#     add_random = ~is_on^fraction # in other words nxor: (is_on and fraction) or (not is_on and not fraction)
#     random_tresholds = firing_tresholds + add_random * np.random.normal(0, temperature * firing_tresholds, len(preds))
#     return (preds >= random_tresholds).astype(int)

def multihot_sample(preds, temperature, firing_loc):  # later implement different variants
    firing_thresholds = np.array(retrieve(firing_loc, "firing_tresholds"))
    firing_fractions = np.array(retrieve(firing_loc, "firing_fractions"))
    fraction = np.random.uniform(0, 1, len(preds)) >= firing_fractions
    is_on = preds >= firing_thresholds
    add_random = ~is_on ^ fraction  # in other words nxor: (is_on and fraction) or (not is_on and not fraction)
    random_tresholds = firing_thresholds + add_random * np.random.normal(0, temperature * firing_thresholds, len(preds))
    return (preds >= random_tresholds).astype(int)


def sample_with_temp(preds, temperature):
    if temperature == 0:
        return np.argmax(preds)
    else:
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(preds), p=preds)


def save_notes_and_durations(store_folder, notes, durations):
    os.makedirs(store_folder, exist_ok=True)
    print(
        f"Saving notes and duration sequences of length {len(notes)} to {truncate_str(str(store_folder), 'prefix', 20)}")
    with open(os.path.join(store_folder, 'notes'), 'wb') as f:
        pkl.dump(notes, f)
    with open(os.path.join(store_folder, 'durations'), 'wb') as f:
        pkl.dump(durations, f)


def retrieve(store_folder, name, extension=False, nonexistent=None):
    if extension:
        name = f"{name}.pickle"
    path_ = os.path.join(store_folder, name)
    if not os.path.exists(path_):
        if nonexistent is None:
            raise Exception(f"Invalid path {store_folder} for the {name} object")
        else:
            return nonexistent
    with open(path_, 'rb') as f:
        object_ = pkl.load(f)
    return object_


def retrieve_distincts_and_lookups(store_folder):
    return retrieve(store_folder, "distincts"), retrieve(store_folder, "lookups")


def retrieve_notes_and_durations(store_folder, **kwargs):
    return retrieve(store_folder, "notes", **kwargs), retrieve(store_folder, "durations", **kwargs)


def retrieve_network_input_output(store_folder, n_if_shortened=None):
    in_, out_ = retrieve(store_folder, "network_input"), retrieve(store_folder, "network_output")
    if n_if_shortened is not None:
        if n_if_shortened < 1:
            n_if_shortened = int(n_if_shortened * len(in_[0]))
        in_ = [x[:n_if_shortened] for x in in_]
        out_ = [x[:n_if_shortened] for x in out_]
    return (in_, out_)


def color_list(kind, mod=None):
    if type(kind) is int:
        kind = np.arange(kind)
    if mod is None:
        mod = len(kind)
    color_map = np.random.rand(mod, 3)
    colors = [0]
    for i in range(1, len(kind)):
        if kind[i] == kind[i - 1]:
            colors.append(colors[-1])
        else:
            colors.append((colors[-1] + 1) % mod)
    return [color_map[i] for i in colors]


def extract_name_from_python_file():
    filename_splitted = ntpath.basename(sys.argv[0]).split(".")
    if len(filename_splitted) > 2:
        raise Exception("The filename contains '.'")
    return filename_splitted[0]


def normalize_array(arr):
    return arr / np.maximum(1, np.tile(np.expand_dims(np.sum(arr, axis=1), axis=1), (1, arr.shape[1])))


def dump(dir, name, file, extension=False):
    if extension:
        name = f"{name}.pickle"
    with open(os.path.join(dir, name), "wb") as f:
        pkl.dump(file, f)


def save_train_test_split(store_folder, io_files=None):
    in_, out_ = retrieve_network_input_output(store_folder) if not io_files else io_files
    dataset_len = len(in_[0])
    trainable = np.ones(dataset_len)
    trainable[:int(0.2 * dataset_len)] = 0
    trainable = trainable.astype(bool)
    np.random.shuffle(trainable)
    train_dir = os.path.join(store_folder, "train")
    test_dir = os.path.join(store_folder, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for suffix, ds in zip(("input", "output"), [in_, out_]):
        train_data = [data[trainable] for data in ds]
        test_data = [data[~trainable] for data in ds]
        name = "network_" + suffix
        dump(train_dir, name, train_data)
        dump(test_dir, name, test_data)
    if not io_files:
        os.remove(os.path.join(store_folder, "network_input"))
        os.remove(os.path.join(store_folder, "network_output"))


def print_to_file(str_, file, indent_level=0):
    str_ = " " * indent_level * 4 + str(str_)
    print(str_, file=file)
    print(str_)


def truncate_str(str_, where="suffix", how_much=10):
    if len(str_) < how_much + 4:
        return str_
    if where == "suffix":
        return re.sub(r'^(.{' + str(how_much) + r'}).*$', '\g<1>...', str_)
    elif where == "prefix":
        return re.sub(r'^.*(.{' + str(how_much) + r'})$', '...\g<1>', str_)
    elif where == "middle":
        pref_len = how_much // 2
        suf_len = how_much - pref_len
        return re.sub(r'^(.{' + str(pref_len) + r'}).*(.{' + str(suf_len) + r'})$', '\g<1>...\g<2>', str_)
    else:
        raise ValueError("Invalid value of the 'where' parameter")


def save_model_to_json(model, dir, name="model"):
    with open(os.path.join(dir, name + ".json"), "w") as f:
        f.write(model.to_json())


def retrieve_final_model(dir):
    model_dir = os.path.join(dir, "final_model")
    return keras.models.load_model(model_dir) if os.path.exists(model_dir) else None


def retrieve_model_from_json(dir, name="model", custom_objects=None):
    with open(os.path.join(dir, name + ".json"), "r") as f:
        if custom_objects is None:
            return model_from_json(f.read())
        else:
            return model_from_json(f.read(), custom_objects)


def retrieve_attention_model(model):
    try:
        return Model(inputs=model.inputs, outputs=[model.get_layer("activation").output])
    except ValueError:
        print("The model has no 'attention' layer. Returning None in retrieve_attention_model")
        return None


def retrieve_best_model(dir, weights_file=None, lambdas=None):
    final_model = retrieve_final_model(dir)
    model = final_model if final_model is not None else retrieve_model_from_json(dir, custom_objects=lambdas)
    weights_file = glob(os.path.join(os.path.join(dir, "weights"), "weights-improvement*"))[-1] \
        if weights_file is None else os.path.join(os.path.join(dir, "weights"), weights_file)
    if not os.path.exists(weights_file):
        raise ValueError(f"Invalid name of the weights file: {weights_file}")
    model.load_weights(weights_file)
    return retrieve_attention_model(model), model, os.path.basename(weights_file)


def print_dict(dict_, print):
    for (key, value) in dict_.items():
        print(f"{key:<5}{value}")


def visualize_model(model_dir, name=None, **params):
    _, m, _ = retrieve_best_model(model_dir)
    if name is None:
        name = "model" + \
               ("_with_shapes" if params.get("show_shapes", False) else "") + \
               ("_with_dtype" if params.get("show_dtype", False) else "")
    plot_model(m, os.path.join(model_dir, f"{name}.png"), **params)
    plot_model(m, os.path.join(model_dir, f"{name}_no_layer_names.png"), **params, show_layer_names=False)
    plot_model(m, os.path.join(model_dir, f"{name}_horizontal.png"), **params, rankdir="LR")


def save_fig(plt, store_dir, name):
    plt.savefig(os.path.join(store_dir, name))
    pdf_dir = os.path.join(store_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    plt.savefig(os.path.join(pdf_dir, f"{name}.pdf"))


def par_dir(dir):
    dir = str(dir).rstrip("/").rstrip("\\")
    return os.path.dirname(dir)


def change_path(path, new_root, new_subfolder, new_extension, new_name=None):
    old_path = Path(path)
    new_path = Path(new_root).joinpath(new_subfolder)
    os.makedirs(new_path, exist_ok=True)
    new_extension = new_extension if new_extension.startswith(".") else "." + new_extension
    new_path = new_path.joinpath(old_path.name) if new_name is None else new_path.joinpath(new_name)
    new_path = new_path.with_suffix(new_extension)
    return new_path


def test_change_path():
    song = "utilities/example/billie_jean.mid"
    new_root = "utilities/example/"
    new_subfolder = "wav"
    new_extension = "mp3"
    print(change_path(song, new_root, new_subfolder, new_extension))
    print(change_path(song, new_root, new_subfolder, new_extension, "new_name"))

def draw_representatives(db, n = 10):
    elements = np.array(glob(os.path.join(db, "*")))
    if len(elements) < n:
        raise Exception(f"There is not enough elements in the database: {db}")
    repr_dir = os.path.join(db, "representatives")
    os.makedirs(repr_dir, exist_ok=True)
    np.random.shuffle(elements)
    representatives = elements[:n]
    for repr_path in representatives:
        new_path = os.path.join(repr_dir, ntpath.basename(repr_path))
        shutil.copyfile(repr_path, new_path)

def frac(a, b):
    return r"\frac{" + str(int(a)) + "}{" + str(int(b)) + "}"

def transform_duration_tex(d):
    if "/" in str(d):
        fr = str(d).split("/")
        d = frac(fr[0], fr[1])
    elif str(d).endswith(".25") or str(d).endswith(".75"):
        d = frac(4*d, 4)
    elif str(d).endswith(".5"):
        d = frac(2*d, 2)
    else:
        d = int(d)
    return fr"${d}$"

def transform_note_tex(n):
     return n.replace(".", " ").replace("#", r"$\sharp$").replace("-", r"$\flat$")

##TODO Question how to sample with temperature from multihot encoding, przygotuj inne pytania na spotkanie

## Zrob jakies unit tests, np czy durations z rests sie zgadzaja
## uporzadkuj code base

def draw_evaluation_model_representatives(model = "random_generator"):
    draw_representatives(f"../evaluation_models/compose/{model}/mp3")

def visualize_evaluated_models():
    visualize_model("../run/two_datasets_attention_hpc/21")
    visualize_model("../run/two_datasets_multihot/08")

if __name__ == "__main__":
    visualize_evaluated_models()
    # draw_evaluation_model_representatives()
    # for filename in glob("../run/two_datasets_multihot/00/weights/weights-improvement*"):
    #     print(filename)
    # describe_dataset("../run/two_datasets_attention/store/version_0")
