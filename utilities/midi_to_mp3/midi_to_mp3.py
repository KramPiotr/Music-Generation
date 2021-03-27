from midi2audio import FluidSynth
from utilities.utils import change_path
from pathlib import Path
import pyaudio  # audio recording
import wave  # file saving
import pygame  # midi playback
import fnmatch  # name matching
import os  # file listing
from icecream import ic
import subprocess
from glob import glob
from tqdm import tqdm

def convert_midi(midi_path, output_dir, new_name = None):
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

    subprocess.call(['fluidsynth', '-ni', sound_font, midi_absolute, '-F', wav_absolute])#, '-T', 'wav', '-a', 'waveout'])#, '-r', str(sample_rate)])
    os.system('ffmpeg -i ' + wav_absolute + ' -y -f mp3 -vbr 1 -ac ' + str(channels) + ' -vn ' + mp3_absolute)
    # os.system('ffmpeg -i ' + wav_absolute + ' -y -f mp3 -ab ' + str(mp3_bitrate) + 'k -ac ' + \
    #           str(channels) + ' -vn ' + mp3_absolute)
              #str(channels) +  '-ar ' + str(sample_rate) + ' -vn ' + mp3_absolute)

def test_convert_midi():
    song = "../example/output_init_None.mid"
    new_root = "utilities/example/"
    convert_midi(song, new_root, "vbred")

def convert_database(db_path):
    songs = glob(os.path.join(db_path, "output*/output*"))
    if len(songs) == 0:
        raise Exception(f"The specified database: {db_path} is empty.")
    for song in tqdm(songs):
        parent = Path(song).parent.name
        print(f"Converting {parent}")
        convert_midi(song, db_path, parent)

def test_convert_database():
    db_path = "../../run/two_datasets_attention/03/compose"
    convert_database(db_path)

if __name__ == "__main__":
    #test_convert_midi()
    test_convert_database()