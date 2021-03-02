import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from utilities.run_utils import set_run_id
from utilities.utils import get_distinct, retrieve, retrieve_network_input_output, print_to_file, save_model_to_json
import shutil

def run(section, dataset_version, create_network, network_params, epochs=100, patience=10, evaluate=True, generate=True, force_run_id=None, next_run=True, trial_run=False, descr=None):
    '''
    Run the model

    :param section: 'dataset'_'model', this representation is used to create directories inside the 'run' directory
    :param dataset_version: number indicating the dataset's version
    :param create_network: function used to create and compile the model
    :param network_params: dictionary of arguments and values applied to create_network
    :param generate: generate music after training the model
    :param force_run_id:
        possible values:
         - None - continue with the current run_id
         - Integer - set run_id to this value but don't update the run_id file
         - 'reset' - set run_id to 0 and update the file
         - 'resetX' - set run_id to x and update the file
    :param next_run: increment run_id when saving to a file or not
    :param trial_run: is it a trial (shortened input) or a full-blown execution
    :param descr: short description of the model
    :return:
    '''
    #force_run_id = 'reset1' #None #Integer #'reset' #'resetX'
    #next_run = True
    n_if_shortened = 100 if trial_run else None

    # data params
    # seq_len = 32
    # embed_size = 100
    # rnn_units = 256
    # use_attention = True

    # run params
    #section = 'two_datasets_attention'
    #dataset_version = 1

    run_dir = os.path.join("..", "run")
    section_folder = os.path.join(run_dir, section)
    if not os.path.exists(section_folder):
        raise ValueError(f"The specified section doesn't exist in the run directory: {section_folder}")

    run_id = set_run_id(section_folder, force_run_id, next_run)
    run_folder = os.path.join(section_folder, run_id)
    if os.path.exists(run_folder):
        shutil.rmtree(run_folder)

    os.mkdir(run_folder)
    compose_folder = os.path.join(run_folder, 'compose')
    weights_folder = os.path.join(run_folder, 'weights')
    os.mkdir(compose_folder)
    os.mkdir(weights_folder)
    if descr is not None:
        with open(os.path.join(run_folder, "description.txt"), "w") as f:
            print(descr, file=f)

    store_model_folder = os.path.join(section_folder, "store")
    if not os.path.exists(store_model_folder):
        store = "_".join(section.split("_")[:-1]) + "_store"
        store_model_folder = os.path.join(run_dir, store)

    store_model_folder = os.path.join(store_model_folder, f"version_{dataset_version}")
    if not os.path.exists(store_model_folder):
        raise ValueError(f"The specified dataset version is invalid. Directory: {store_model_folder}")

    if section.endswith("multihot"):
        _, n_durations = retrieve(store_model_folder, "distincts")
        n_notes = 12
    else:
        _, n_notes, _, n_durations = retrieve(store_model_folder, "distincts")

    train_folder = os.path.join(store_model_folder, "train")
    network_input, network_output = retrieve_network_input_output(train_folder, n_if_shortened)

    model, att_model = create_network(n_notes, n_durations, **network_params)
    model.summary()
    with open(os.path.join(run_folder, "model.txt"),'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

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
        monitor='val_loss',
        restore_best_weights=True,
        patience = 3
    )

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(run_folder, "logs"), histogram_freq=1),

    csv_logger = CSVLogger(os.path.join(run_folder, 'log.csv'), append=False, separator=';')

    callbacks_list = [
        checkpoint1,
        checkpoint2,
        early_stopping,
        tensorboard,
        csv_logger,
     ]

    history_callback = model.fit(network_input, network_output,
              epochs=epochs, batch_size=32,
              validation_split = 0.2,
              callbacks=callbacks_list,
              shuffle=True,
             )
    model.save_weights(os.path.join(weights_folder, "weights.h5"))
    save_model_to_json(model, run_folder)
    # save_model_to_json(att_model, run_folder, name="att_model")

    if evaluate:
        test_folder = os.path.join(store_model_folder, "test")
        test_input, test_output = retrieve_network_input_output(test_folder, n_if_shortened)
        with open(os.path.join(run_folder, f"test_results.txt"), "w") as f:
            print_to_file(f"Evaluation using a test set containing {len(test_input[0])} sequences", f)
            results = model.evaluate(test_input, test_output, batch_size=32)
            for n, r in zip(model.metrics_names, results):
                print_to_file(f"{n:>13}: {r:.4f}", f)

    if section.endswith("multihot"):
        from RNN_attention_multihot_encoding.model_specific_utils import record_firing
        record_firing(run_folder, os.path.join(store_model_folder, "test"), test=trial_run)

    if generate:
        #TODO predict
        pass



#TODO test the model and save to the file, maybe connect to 'predict' as well

if __name__ == "__main__":
    network_params = {
        "seq_len": 32,
        "embed_size": 100,
        "rnn_units": 256,
        "use_attention": True,
    }
    from RNN_attention_multihot_encoding.model import create_network

    run("two_datasets_multihot", 1, create_network, network_params, trial_run=True, descr="Trial, safe to delete")
    # run("two_datasets_multihot", 1, create_network, network_params, descr="Full model trained with the additional Dense layer")

    # network_params = {
    #     "embed_size": 100,
    #     "rnn_units": 256,
    #     "use_attention": True,
    #     "n_dense": 2,
    # }
    # from RNN_attention.model import create_network
    # run("two_datasets_attention", 1, create_network, network_params, force_run_id='reset4', next_run=True, descr="Full model trained with the additional Dense layer")
