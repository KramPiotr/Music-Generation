import os
import numpy as np
import pickle as pkl
from utilities.utils import retrieve_network_input_output, retrieve_best_model, print_to_file


def record_firing(model_dir, network_io_dir):
    _, model = retrieve_best_model(model_dir)
    test_input, test_output = retrieve_network_input_output(network_io_dir)
    predictions = model.predict(test_input, batch_size=32, verbose=1)
    note_predictions = predictions[0]
    note_truth = test_output[0]
    firing = [[[], []] for _ in range(12)]
    for pr, tr in zip(note_predictions, note_truth):
        for i, (p, t) in enumerate(zip(pr, tr)):
            firing[i][int(t)].append(p)

    firing = [[np.array(d) for d in note_data] for note_data in firing]
    firing_mean_std = [[(np.mean(d), np.std(d)) for d in note_data] for note_data in firing]
    std_tresholds = [np.interp(0, (-dict_[0][1], dict_[1][1]), (dict_[0][0], dict_[1][0])) for dict_ in firing_mean_std]
    note_tresholds= []
    with open(os.path.join(model_dir, "firing_log.txt"), "w") as f:
        for i, note_data in enumerate(firing):
            probabilities = []
            lens = []
            for j, probs in enumerate(note_data):
                if len(probs) == 0:
                    raise Exception(f"Note {i} has 0 occurences of class {j}")
                lens.append(len(probs))
                probabilities.extend([(p, j) for p in probs])
            probabilities.sort()
            max_F_score = 0
            treshold = 0.5
            print_data = (0, 0, 0, 0)
            fn = 0
            for j, (p, t) in enumerate(probabilities[:-1]):
                fn += t
                tp = lens[1] - fn
                fp = len(probabilities) - (j + 1) - tp
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                F1 = 2 * precision * recall / (precision + recall)
                if F1 > max_F_score:
                    max_F_score = F1
                    treshold = p + (probabilities[j+1][0] - p) / 2
                    print_data = (fn, tp, fp, lens[0] - fp)
            note_tresholds.append(treshold)
            print_to_file(f"Note {i}:", f)

            print_to_file(f"treshold: {treshold:.4f}, F1-score: {F1:.2f}, precision: {precision:.2f}, recall: {recall:.2f}", f, indent_level=1)
            print_to_file(f"std treshold (primary idea): {std_tresholds[i]:.4f}", f, indent_level=1)

            print_to_file(f"off: {lens[0]}", f, indent_level=1)
            print_to_file(f"true negatives: {print_data[3]}, fraction: {print_data[3]/lens[0]:.2f}", f, indent_level=2)
            print_to_file(f"false positives: {print_data[2]}", f, indent_level=2)
            print_to_file(f"mean: {firing_mean_std[i][0][0]:.4f}, std: {firing_mean_std[i][0][1]}:.4f", f, indent_level=2)

            print_to_file(f"on: {lens[1]}", f, indent_level=1)
            print_to_file(f"true positives: {print_data[1]}, fraction: {print_data[1]/lens[1]:.2f}", f, indent_level=2)
            print_to_file(f"false negatives: {print_data[0]}", f, indent_level=2)
            print_to_file(f"mean: {firing_mean_std[i][1][0]:.4f}, std: {firing_mean_std[i][1][1]:.4f}", f, indent_level=2)

            print_to_file("", f)

    with open(os.path.join(model_dir, "firing_tresholds"), "wb") as f:
        pkl.dump(note_tresholds, f)



if __name__ == "__main__":
    record_firing("../run/two_datasets_multihot/00/", "../run/two_datasets_store/version_1/test") #TODO test once training finished