from utilities.utils import open_midi, extract_notes


def extract_notes_test():
    seq_len = 32
    original_score = open_midi("datasets/MIREX_dataset/MIDIs/004.mid")
    n, d = extract_notes(original_score, seq_len)

    #parts = instrument.partitionByInstrument(original_score) # leave only notes
    #print(list(zip(parts.parts[0].recurse()[:10], n[seq_len:seq_len+10], d[seq_len:seq_len+10])))


extract_notes_test()