# Author: Piotr Kram
import pickle as pkl
import os

def id_to_str(id):
    return ('0' + str(id)) if id < 10 else str(id)

def set_run_id(store_folder, force_run_id, next_run):
    '''

    :param store_folder: describes the location of the run_id file
    :param force_run_id:
        possible values:
         - None - continue with the current run_id
         - Integer - set run_id to this value but don't update the run_id file
         - 'reset' - set run_id to 0 and update the file
         - 'resetX' - set run_id to x and update the file
    :param next_run: increment run_id when saving to a file or not
    :return:
    '''
    run_id_file = os.path.join(store_folder, "run_id.txt")
    if force_run_id is None:
        with open(run_id_file, 'r') as f:
            run_id = int(f.read())
        with open(run_id_file, 'w') as f:
            f.write(str(run_id + next_run))
    else:
        if type(force_run_id) is not int:
            run_id = 0 if len(force_run_id) < 6 else int(force_run_id[5:])
            with open(run_id_file, 'w') as f:
                f.write(str(run_id + next_run))
        else:
            run_id = force_run_id
    return id_to_str(run_id)