from utilities.utils import retrieve_best_model, retrieve_distincts_and_lookups
import io
import numpy as np

def inspect_embeddings(model_dir = '../run/two_datasets_attention_hpc/21'):
    def in_chord(pitch, chord):
        return pitch in chord.replace(f"{pitch}#", "").replace(f"{pitch}-", "")

    def mean_L2_distance(embeddings):
        distances = []
        for e1 in embeddings:
            for e2 in embeddings:
                distances.append(np.sqrt(sum((e1 - e2)**2)))
        return np.mean(distances)

    att_model, model, weights_file = retrieve_best_model(model_dir, weights_file='weights.h5')
    distincts, lookups = retrieve_distincts_and_lookups('../run/two_datasets_attention/store/version_2')
    (notes_names, n_notes, duration_names, n_durations), (
        note_to_int, int_to_note, duration_to_int, int_to_duration) = distincts, lookups
    note_embedding = {int_to_note[i] : model.layers[2].get_weights()[0][i] for i in range(n_notes)}

    mean_L2_distances = {}
    embeddings = list(note_embedding.values())
    mean_L2_distances['overall'] = mean_L2_distance(embeddings)

    pitches = [key for key in note_to_int.keys() if len(key) < 3 and key != "S"]
    for p in pitches:
        embeddings = [embed for chord, embed in note_embedding.items() if in_chord(p, chord)]
        mean_L2_distances[p] = mean_L2_distance(embeddings)

    #works but problem with G#
    print(5)
    #duration_embedding = model.layers[3].get_weights()[0][0]
    #encoder = info.features['text'].encoder

    #out_v = io.open('vecs.tsv', 'w', encoding='utf-8')

    # for num, word in enumerate(encoder.subwords):
    #   vec = weights[num+1] # skip 0 for padding.
    #   out_v.write(word+'\t'.join([str(x) for x in vec]) + "\n")
    #
    # out_v.close()


if __name__ == "__main__":
    inspect_embeddings()