import os
import pickle

run_id = '00'
run_folder = "../run/MIREX_multihot"
run_folder = os.path.join(run_folder, run_id)
store_folder = os.path.join(run_folder, 'store')

with open(os.path.join(store_folder, 'notes'), 'rb') as f:
    notes = pickle.load(f)
with open(os.path.join(store_folder, 'durations'), 'rb') as f:
    durations = pickle.load(f)
with open(os.path.join(store_folder, 'lookups'), 'rb') as f:
    lookups = pickle.load(f)
with open(os.path.join(store_folder, 'distincts'), 'rb') as f:
    distincts = pickle.load(f)

a = 5
b = 3
print(len(distincts[0])) # notes - 112828, durations - 74
print("gowno")
#print(lookups)