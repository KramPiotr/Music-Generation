import os
import pickle

section = 'MIREX'
run_id = '02'

run_folder = os.path.join("run", section)
run_folder = os.path.join(run_folder, run_id)

store_folder = os.path.join(run_folder, 'store')

with open(os.path.join(store_folder, 'notes'), 'rb') as f:
    notes = pickle.load(f)  # ['G2', 'D3', 'B3', 'A3', 'B3', 'D3', 'B3', 'D3', 'G2',...]
with open(os.path.join(store_folder, 'durations'), 'rb') as f:
    durations = pickle.load(f)
with open(os.path.join(store_folder, 'lookups'), 'rb') as f:
    lookups = pickle.load(f)
with open(os.path.join(store_folder, 'distincts'), 'rb') as f:
    distincts = pickle.load(f)

print(len(distincts[0])) # notes - 112828, durations - 74
#print(lookups)