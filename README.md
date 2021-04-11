# Music-Generation
A repository for the machine learning part of my Part II project

## Setup

1. First install requirements: 
   
   `pip install -r requirements.txt`
   
2. Verify GPU support:
   
   `python ./test_GPU_tensorflow.py`

3. Verify that the ***datasets*** folder is added to the root of the repository

4. Run process datasets function from the ***process_midi_archive.py*** file.

   `python ./utilities/process_midi_archive.py`

5. Run hyperparameter tuning. Results are saved in ***./run/two_datasets_attention/[run id]***. 
   You can control the id under which the results are saved by modifying the ***run_id.txt*** file in this folder, 
   created during the first run of the ***hyperparameter_tuning.py*** file.

   `python ./utilities/hyperparameter_tuning.py`

6. To gather the results from multiple runs in one, ***results_analysis.txt*** file you can run the ***results_analysis.py*** file.

   `python ./utilities/results_analysis.py`