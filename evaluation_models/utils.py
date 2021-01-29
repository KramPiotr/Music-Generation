import os
from utilities.utils import extract_name_from_python_file

def create_output_folder():
    script_name = extract_name_from_python_file()
    compose_path = os.path.join("compose", script_name)
    os.makedirs(compose_path, exist_ok=True)
    return compose_path