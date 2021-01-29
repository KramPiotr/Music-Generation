from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import note_seq
import os
import numpy as np
from evaluation_models.utils import create_output_folder
from tqdm import tqdm

compose_path = create_output_folder()

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
length = 50

music_vae = TrainedModel(
      configs.CONFIG_MAP['hierdec-mel_16bar'],
      batch_size=1,
      checkpoint_dir_or_path='hierdec-mel_16bar.tar')


temps = np.linspace(0.1, 1.5, 50)

for i, t in tqdm(enumerate(temps)):
      generated_sequences = music_vae.sample(n=1, length=length, temperature=t)
      for ns in generated_sequences:
            note_seq.sequence_proto_to_midi_file(ns, os.path.join(compose_path, f'output{i}_temp-{t:.2f}.mid'))

#DEACTIVATE VENV