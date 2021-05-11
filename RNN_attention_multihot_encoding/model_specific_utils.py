import os
import numpy as np
import pickle as pkl
from utilities.utils import retrieve_network_input_output, retrieve_best_model, print_to_file
import matplotlib.pyplot as plt
import seaborn as sns


def record_firing(model_dir, network_io_dir, io_fragment=None, suffix="", test=False):
    _, model, _ = retrieve_best_model(model_dir)  # discard nonexistent attention model and path to the weights file
    test_input, test_output = retrieve_network_input_output(
        network_io_dir) if io_fragment is None else retrieve_network_input_output(network_io_dir,
                                                                                  io_fragment)  # io_fragment is applied if we want to use only a fracture of data e.g. in trial runs
    firing = [[[], []] for _ in range(12)]  # to record for each note firing when the note is on and off
    if test:  # for test randomize the tresholds
        for i in range(12):
            for j in range(2):
                l = np.random.randint(100, 300)
                low = 0 + j * 0.2
                firing[i][j] = np.random.uniform(low, low + 0.5, l)
    else:
        predictions = model.predict(test_input, batch_size=32, verbose=1)
        note_predictions = predictions[0]
        note_truth = test_output[0]
        for pr, tr in zip(note_predictions, note_truth):
            for i, (p, t) in enumerate(zip(pr, tr)):
                firing[i][int(t)].append(p)  # i is the note, t is the true firing (0 or 1), p is the predicted firing

    firing = [[np.array(d) for d in note_data] for note_data in
              firing]  # convert firings into numpy array. Doesn't change the structure of firing
    firing_mean_std = [[(np.mean(d), np.std(d)) for d in note_data] for note_data in
                       firing]  # firing_mean_std[note][true label] = (mean firing, std of firing)
    std_tresholds = [np.interp(0, (-dict_[0][1], dict_[1][1]), (dict_[0][0], dict_[1][0])) for dict_ in
                     firing_mean_std]  # interpolate between the firing means depending on the length of the std
    pcs_names = np.linspace(0, 100, 1001)  # percentile values to sample
    percentiles = [[np.percentile(d, pcs_names) for d in note_data] for note_data in
                   firing]  # precentiles[note][true label] = sorted percentile values

    note_tresholds = []
    note_on_fractions = []

    cmap = plt.get_cmap('tab20')
    ind = np.arange(9) * 2
    ind = np.append(ind, [1, 3, 5])  # colour indices for the colormap

    firing_dir_name = "firing" + (("_" + suffix) if len(suffix) > 0 else "")
    firing_dir = os.path.join(model_dir, firing_dir_name)
    os.makedirs(firing_dir, exist_ok=True)

    with open(os.path.join(firing_dir, "firing_log.txt"), "w") as f:
        for i, note_data in enumerate(firing):  # i is a note, note_data are firings from the labels
            probabilities = []
            lens = []
            for j, probs in enumerate(note_data):  # j is the true label, probs are the firings
                if len(probs) == 0:
                    raise Exception(f"Note {i} has 0 occurences of class {j}")
                lens.append(len(probs))  # lens contains the number of firings from the true label
                probabilities.extend([(p, j) for p in probs])
            probabilities.sort()  # probabilities is a sorted list of tuples (firing, true label)
            max_F_score = 0
            treshold = 0.5
            print_data = (0, 0, 0, 0)
            print_pr_rc = (0, 0)
            fn = 0  # false negatives
            precisions = []
            recalls = []

            for j, (p, t) in enumerate(probabilities[:-1]):
                fn += t
                tp = lens[1] - fn  # true positivess
                fp = len(probabilities) - (j + 1) - tp
                precision = tp / max(1, (tp + fp))
                precisions.append(precision)
                recall = tp / max(1, (tp + fn))
                recalls.append(recall)
                if precision + recall < 0.000001:
                    F1 = 0
                else:
                    F1 = 2 * precision * recall / (precision + recall)  # min(FN + FP), localised approach
                if F1 > max_F_score:
                    max_F_score = F1
                    treshold = p + (probabilities[j + 1][0] - p) / 2
                    print_data = (fn, tp, fp, lens[0] - fp)
                    print_pr_rc = (precision, recall)
            note_tresholds.append(treshold)
            note_on_fractions.append(lens[1] / (lens[0] + lens[1]))
            print_to_file(f"Note {i}:", f)

            print_to_file(
                f"treshold: {treshold:.4f}, F1-score: {max_F_score:.2f}, precision: {print_pr_rc[0]:.2f}, recall: {print_pr_rc[1]:.2f}",
                f, indent_level=1)
            print_to_file(f"std treshold (primary idea): {std_tresholds[i]:.4f}", f, indent_level=1)

            print_to_file(f"off: {lens[0]}", f, indent_level=1)
            print_to_file(f"true negatives: {print_data[3]}, fraction: {print_data[3] / lens[0]:.2f}", f,
                          indent_level=2)
            print_to_file(f"false positives: {print_data[2]}", f, indent_level=2)
            print_to_file(f"mean: {firing_mean_std[i][0][0]:.4f}, std: {firing_mean_std[i][0][1]:.4f}", f,
                          indent_level=2)

            print_to_file(f"on: {lens[1]}", f, indent_level=1)
            print_to_file(f"true positives: {print_data[1]}, fraction: {print_data[1] / lens[1]:.2f}", f,
                          indent_level=2)
            print_to_file(f"false negatives: {print_data[0]}", f, indent_level=2)
            print_to_file(f"mean: {firing_mean_std[i][1][0]:.4f}, std: {firing_mean_std[i][1][1]:.4f}", f,
                          indent_level=2)

            print_to_file("", f)
            plt.plot(recalls, precisions, color=cmap(ind[i]), lw=0.5, label=f"Note {i}")  # lw - line width
            plt.plot(print_pr_rc[1], print_pr_rc[0], 'bo')  # 'bo' for blue dot
        plt.plot([0, 1], [1, 0], color='black')
        plt.legend()  # loc='upper right')
        plt.title("Precision-recall curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(os.path.join(firing_dir, "firing_PR_curve"))

    with open(os.path.join(firing_dir, "firing_tresholds"), "wb") as f:
        pkl.dump(note_tresholds, f)

    with open(os.path.join(firing_dir, "firing_fractions"), "wb") as f:
        pkl.dump(note_on_fractions, f)

    with open(os.path.join(firing_dir, "firing_percentiles"), "wb") as f:
        pkl.dump(percentiles, f)

    with open(os.path.join(firing_dir, "percentile_names"), "wb") as f:
        pkl.dump(pcs_names, f)

    # PR curves depending on notes
    dist_dir = os.path.join(firing_dir, "distributions")
    os.makedirs(dist_dir, exist_ok=True)
    for i in range(12):
        plt.figure()
        colors = ['blue', 'red']
        for j in range(2):
            sns.distplot(firing[i][j],
                         bins=1000,
                         kde=False,
                         color=colors[j],
                         hist_kws={"linewidth": 15, 'alpha': 0.5})
        plt.xlabel('Prediction')
        plt.ylabel('Count')
        plt.yscale("log")
        plt.xlim(0, 1)
        plt.title(f"Probability density function for note {i}")
        plt.savefig(os.path.join(dist_dir, f"Note_{i}_PDF"))


def test_firing():
    record_firing("../run/two_datasets_multihot/08/", "../run/two_datasets_store/version_3/test")
    record_firing("../run/two_datasets_multihot/08/", "../run/two_datasets_store/version_3/train", suffix="train")
    # record_firing("../run/two_datasets_multihot/09/", "../run/two_datasets_store/version_3/test")


if __name__ == "__main__":
    # cmap = plt.get_cmap('tab20')
    # for i in range(20):
    #     print(cmap(i))
    # np.random.seed(0)
    # record_firing("../run/two_datasets_multihot/01/", "../run/two_datasets_store/version_1/train", io_fragment=0.6,
    #               suffix="train")  # TODO test once training finished
    test_firing()
