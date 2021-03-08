from glob import glob
import pandas as pd
import os
for i, result in enumerate(glob("..\\run\\two_datasets_attention_hpc\\*")):
    test_results = os.path.join(result, "test_results.txt")
    log_csv = os.path.join(result, "log.csv")
    if not os.path.exists(test_results) or not os.path.exists(log_csv):
        continue
    print(f"\n\n{i}. Results {result}\n")
    with open(test_results, "r") as f:
        print(f.read())
    log_csv = pd.read_csv(log_csv, delimiter=";")
    print(f"Best result for validation, overall loss".ljust(50) + f"{min(log_csv['val_loss'])}")
    print(f"Best result for validation, pitch loss".ljust(50) + f"{min(log_csv['val_pitch_loss'])}")
    print(f"Best result for validation, duration loss".ljust(50) + f"{min(log_csv['val_duration_loss'])}\n")

    print(f"Best result for training, overall loss".ljust(50) + f"{min(log_csv['loss'])}")
    print(f"Best result for training, pitch loss".ljust(50) + f"{min(log_csv['pitch_loss'])}")
    print(f"Best result for training, duration loss".ljust(50) + f"{min(log_csv['duration_loss'])}")