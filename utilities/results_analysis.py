from glob import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from utilities.utils import print_to_file, save_fig
from functools import partial

def plot_results(results_database):
    scores = []
    for i, result in enumerate(glob(os.path.join(results_database, "*"))):
        test_results = os.path.join(result, "test_results.txt")
        if not os.path.exists(test_results):
            continue
        with open(test_results, "r") as f:
            scores.append((float(f.readlines()[1].split(":")[1]), result[-2:]))
    scores.sort()
    losses = [x[0] for x in scores]
    plt.bar(np.arange(len(losses)), losses)
    plt.title("Distribution for the test loss in the tuned models")
    plt.xlabel("Models")
    plt.ylabel("Test loss")
    plt.yticks([])
    save_fig(plt, results_database, "test_loss")

def analyse_results(results_database):
    with open(os.path.join(results_database, "results_analysis.txt"), "w") as f:
        print_ = partial(print_to_file, file=f)
        for i, result in enumerate(glob(os.path.join(results_database, "*"))):
            test_results = os.path.join(result, "test_results.txt")
            log_csv = os.path.join(result, "log.csv")
            if not os.path.exists(test_results) or not os.path.exists(log_csv):
                continue
            print_(f"\n\n{i}. Results {result}\n")
            with open(test_results, "r") as f:
                print_(f.read())
            log_csv = pd.read_csv(log_csv, delimiter=";")
            print_(f"Best result for validation, overall loss".ljust(50) + f"{min(log_csv['val_loss'])}")
            print_(f"Best result for validation, pitch loss".ljust(50) + f"{min(log_csv['val_pitch_loss'])}")
            print_(f"Best result for validation, duration loss".ljust(50) + f"{min(log_csv['val_duration_loss'])}\n")

            print_(f"Best result for training, overall loss".ljust(50) + f"{min(log_csv['loss'])}")
            print_(f"Best result for training, pitch loss".ljust(50) + f"{min(log_csv['pitch_loss'])}")
            print_(f"Best result for training, duration loss".ljust(50) + f"{min(log_csv['duration_loss'])}")

def analyse_hpc_results():
    results_database = "..\\run\\two_datasets_attention_hpc\\"
    analyse_results(results_database)

def plot_hpc_results():
    results_database = "..\\run\\two_datasets_attention_hpc\\"
    plot_results(results_database)

def analyse_attention_models():
    results_database = "..\\run\\two_datasets_attention"
    analyse_results(results_database)

def analyse_multihot_models():
    results_database = "..\\run\\two_datasets_multihot"
    analyse_results(results_database)

if __name__ == "__main__":
    plot_hpc_results()