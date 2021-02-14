import os
import numpy as np
import pickle as pkl
from utilities.utils import retrieve_network_input_output, retrieve_best_model, print_to_file
import matplotlib.pyplot as plt
import seaborn as sns


def record_firing(model_dir, network_io_dir, io_fragment = None, suffix="", test=False):
    _, model, _ = retrieve_best_model(model_dir)
    test_input, test_output = retrieve_network_input_output(network_io_dir) if io_fragment is None else retrieve_network_input_output(network_io_dir, io_fragment)
    firing = [[[], []] for _ in range(12)]
    if test:
        for i in range(12):
            for j in range(2):
                l = np.random.randint(100, 300)
                low = 0 + j * 0.2
                firing[i][j] = np.random.uniform(low, low+0.5, l)
    else:
        predictions = model.predict(test_input, batch_size=32, verbose=1)
        note_predictions = predictions[0]
        note_truth = test_output[0]
        for pr, tr in zip(note_predictions, note_truth):
            for i, (p, t) in enumerate(zip(pr, tr)):
                firing[i][int(t)].append(p)

    firing = [[np.array(d) for d in note_data] for note_data in firing]
    firing_mean_std = [[(np.mean(d), np.std(d)) for d in note_data] for note_data in firing]
    std_tresholds = [np.interp(0, (-dict_[0][1], dict_[1][1]), (dict_[0][0], dict_[1][0])) for dict_ in firing_mean_std]
    pcs_names = np.linspace(0, 100, 1001)
    percentiles = [[np.percentile(d, pcs_names) for d in note_data] for note_data in firing]
    note_tresholds= []
    note_on_fractions = []
    cmap = plt.get_cmap('tab20')
    ind = np.arange(9) * 2
    ind = np.append(ind, [1, 3, 5])
    firing_dir = os.path.join(model_dir, f"firing_{suffix}")
    os.makedirs(firing_dir, exist_ok=True)
    with open(os.path.join(firing_dir, "firing_log.txt"), "w") as f:
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
            print_pr_rc = (0, 0)
            fn = 0
            precisions = []
            recalls = []

            for j, (p, t) in enumerate(probabilities[:-1]):
                fn += t
                tp = lens[1] - fn
                fp = len(probabilities) - (j + 1) - tp
                precision = tp / (tp + fp)
                precisions.append(precision)
                recall = tp / (tp + fn)
                recalls.append(recall)
                F1 = 2 * precision * recall / (precision + recall) #min(FN + FP), localised approach
                if F1 > max_F_score:
                    max_F_score = F1
                    treshold = p + (probabilities[j+1][0] - p) / 2
                    print_data = (fn, tp, fp, lens[0] - fp)
                    print_pr_rc = (precision, recall)
            note_tresholds.append(treshold)
            note_on_fractions.append(lens[1]/(lens[0] + lens[1]))
            print_to_file(f"Note {i}:", f)

            print_to_file(f"treshold: {treshold:.4f}, F1-score: {max_F_score:.2f}, precision: {print_pr_rc[0]:.2f}, recall: {print_pr_rc[1]:.2f}", f, indent_level=1)
            print_to_file(f"std treshold (primary idea): {std_tresholds[i]:.4f}", f, indent_level=1)

            print_to_file(f"off: {lens[0]}", f, indent_level=1)
            print_to_file(f"true negatives: {print_data[3]}, fraction: {print_data[3]/lens[0]:.2f}", f, indent_level=2)
            print_to_file(f"false positives: {print_data[2]}", f, indent_level=2)
            print_to_file(f"mean: {firing_mean_std[i][0][0]:.4f}, std: {firing_mean_std[i][0][1]:.4f}", f, indent_level=2)

            print_to_file(f"on: {lens[1]}", f, indent_level=1)
            print_to_file(f"true positives: {print_data[1]}, fraction: {print_data[1]/lens[1]:.2f}", f, indent_level=2)
            print_to_file(f"false negatives: {print_data[0]}", f, indent_level=2)
            print_to_file(f"mean: {firing_mean_std[i][1][0]:.4f}, std: {firing_mean_std[i][1][1]:.4f}", f, indent_level=2)

            print_to_file("", f)
            plt.plot(recalls, precisions, color=cmap(ind[i]), lw=0.5, label=f"Note {i}")
            plt.plot(print_pr_rc[1], print_pr_rc[0], 'bo')
        plt.plot([0, 1], [1, 0], color='black')
        plt.legend()#loc='upper right')
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





if __name__ == "__main__":
    # cmap = plt.get_cmap('tab20')
    # for i in range(20):
    #     print(cmap(i))
    np.random.seed(0)
    record_firing("../run/two_datasets_multihot/01/", "../run/two_datasets_store/version_1/train", io_fragment=0.6, suffix="train") #TODO test once training finished