import pickle as pkl
import os

def id_to_str(id):
    return ('0' + str(id)) if id < 10 else str(id)

def set_run_id(cwd, force_run_id, next_run):
    '''

    :param cwd: describes the location of the run_id file
        should be os.getcwd()
    :param force_run_id:
        possible values:
         - None - continue with the current run_id
         - Integer - set run_id to this value but don't update the run_id file
         - 'reset' - set run_id to 0 and update the file
         - 'resetX' - set run_id to x and update the file
    :param next_run:
    :return:
    '''
    run_id_file = os.path.join(cwd, "run_id")
    if not force_run_id:
        with open(run_id_file, 'rb') as f:
            run_id = pkl.load(f)
        if next_run:
            run_id += 1
        with open(run_id_file, 'wb') as f:
            pkl.dump(run_id, f)
    else:
        if force_run_id[:5] == "reset":
            run_id = 0 if len(force_run_id) < 6 else int(force_run_id[5:])
            with open(run_id_file, 'wb') as f:
                pkl.dump(run_id, f)
        else:
            run_id = force_run_id
    return id_to_str(run_id)