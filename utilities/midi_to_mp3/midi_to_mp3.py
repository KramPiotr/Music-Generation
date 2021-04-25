from utilities.utils import change_path
from utilities.run_utils import id_to_str
from pathlib import Path
import os  # file listing
import subprocess
from glob import glob
from tqdm import tqdm


def convert_midi(midi_path, output_dir, new_name=None):
    # sample_rate = 44100  # Sample rate used for WAV/MP3
    channels = 1  # Audio channels (1 = mono, 2 = stereo)
    # mp3_bitrate = 128  # Bitrate to save MP3 with in kbps (CBR)

    if not os.path.exists(midi_path):
        raise ValueError(f"Invalid midi path: {midi_path}")
    mp3_path = change_path(midi_path, output_dir, "mp3", "mp3", new_name)
    wav_path = change_path(midi_path, output_dir, "wav", "wav", new_name)
    midi_absolute = str(Path(midi_path).absolute())
    wav_absolute = str(wav_path.absolute())
    mp3_absolute = str(mp3_path.absolute())

    sound_font = str(Path("GeneralUser GS v1.471.sf2").absolute())

    # from mido import MidiFile
    # mid = MidiFile(midi_absolute)
    # print(mid.length)

    subprocess.call(['fluidsynth', '-ni', sound_font, midi_absolute, '-F',
                     wav_absolute])  # , '-T', 'wav', '-a', 'waveout'])#, '-r', str(sample_rate)])
    os.system('ffmpeg -i ' + wav_absolute + ' -y -f mp3 -vbr 1 -ac ' + str(channels) + ' -vn ' + mp3_absolute)
    # os.system('ffmpeg -i ' + wav_absolute + ' -y -f mp3 -ab ' + str(mp3_bitrate) + 'k -ac ' + \
    #           str(channels) + ' -vn ' + mp3_absolute)
    # str(channels) +  '-ar ' + str(sample_rate) + ' -vn ' + mp3_absolute)


def test_convert_midi():
    song = "../example/output_init_None.mid"
    new_root = "utilities/example/"
    convert_midi(song, new_root, "vbred")


def convert_database(db_path=None, date_prefix=None, section=None, version_id=None):
    if db_path is None:
        if section is None or version_id is None:
            raise ValueError("Unspecified section or version_id.")
        else:
            db_path = get_db_path(section, version_id)
    prefix = "output" + ("" if date_prefix is None else ("-" + date_prefix)) + "*"
    songs = glob(os.path.join(db_path, f"{prefix}/output*"))
    if len(songs) == 0 and date_prefix is None:
        raise Exception(f"The specified database: {db_path} is empty.")
    print(f"Found {len(songs)} songs matching the prefix {prefix} of the database {db_path}")
    for song in tqdm(songs):
        parent = Path(song).parent.name
        print(f"Converting {parent}")
        convert_midi(song, db_path, parent)


# def get_db_path(section, version_id, level=1):
#     if __name__ == "__main__":
#         level = 2
#     return "../" * level + f"run/{section}/{id_to_str(version_id)}/compose"

def get_db_path(section, version_id):
    return f"../../run/{section}/{id_to_str(version_id)}/compose"


def convert_attention_database(date_prefix=None):
    db_path = "../../run/two_datasets_attention/03/compose"
    convert_database(db_path, date_prefix=date_prefix)

def convert_databases():
    convert_database(section="two_datasets_multihot", version_id=8, date_prefix="2021_04_17")
    convert_database(section="two_datasets_attention_hpc", version_id=21, date_prefix="2021_04_17")


def convert_simple_database(db, output_prefix="output", date_prefix=None):
    prefix = output_prefix + ("" if date_prefix is None else ("-" + date_prefix)) + "*"
    songs = glob(os.path.join(db, prefix))
    if len(songs) == 0 and date_prefix is None:
        raise Exception(f"The specified database: {db} is empty.")
    print(f"Found {len(songs)} songs matching the prefix {prefix} of the database {db}")
    for song in tqdm(songs):
        convert_midi(song, db)


def convert_evaluation_models_database(date_prefix=None):
    '''
    :param date_prefix: in the format [Year]_[Month]_[Date]--[Hour]_[Minute]_[Second]
    :return:
    '''
    db_path = "../../evaluation_models/compose/*"
    db_glob = list(glob(db_path))
    for i, db in enumerate(db_glob):
        print(f"{i + 1}/{len(db_glob)} Converting {db}")
        convert_simple_database(db, date_prefix=date_prefix)


def convert_MM_database():
    db_path = "../../evaluation_models/compose/MM_generator"
    convert_simple_database(db_path)

def convert_Bach(prefix = None):
    db_path = "../../evaluation_models/Bach"
    if prefix is None:
        convert_simple_database(db_path, output_prefix="")
    else:
        convert_simple_database(db_path, output_prefix=prefix)

def convert_Chopin():
    db_path = "../../evaluation_models/Chopin"
    convert_simple_database(db_path, output_prefix="chpn")


if __name__ == "__main__":
    # convert_Chopin()
    convert_Bach()
    # convert_databases()
    # convert_database(get_db_path("two_datasets_multihot", 1))
    # convert_database(get_db_path("two_datasets_attention_hpc", 21))
    # convert_evaluation_models_database(date_prefix="2021_04_17")
    # test_convert_midi()
    # test_convert_database()
    # convert_MM_database()
