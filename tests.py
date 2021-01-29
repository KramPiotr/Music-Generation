from utilities.utils import open_midi, extract_notes


def extract_notes_test():
    seq_len = 32
    original_score = open_midi("datasets/MIREX_dataset/MIDIs/004.mid")
    n, d = extract_notes(original_score, seq_len)

    #parts = instrument.partitionByInstrument(original_score) # leave only notes
    #print(list(zip(parts.parts[0].recurse()[:10], n[seq_len:seq_len+10], d[seq_len:seq_len+10])))


extract_notes_test()

#######################################################3plagiarism detection

# for seq in tqdm(itertools.product([0, 1], repeat=24)):
#     chords = (seq[:12], seq[12:24])
#     iou = IoU(np.array(chords[0]), np.array(chords[1]))
#     if iou > 0:
#         iou_scores[chords] = iou
#######################################################3plagiarism detection

# a = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1])
# b = np.array([1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1])
# c = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# d = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#######################################################3plagiarism detection
# a = np.array(list(iou_scores.keys())[0])
# b = np.array(list(iou_scores.keys())[28])
# c = np.array(list(iou_scores.keys())[9])
# d = np.array(list(iou_scores.keys())[123])
#
# IoU(a, b)
# IoU(a, c)
# IoU(a, d)
# IoU(b, c)
# IoU(b, d)
# IoU(c, d)
#
# aa = tuple(a)
# bb = tuple(b)
# cc = tuple(c)
# dd = tuple(d)
#
# print("-" * 50)
#
# IoU(aa, bb, iou_scores)
# IoU(aa, cc, iou_scores)
# IoU(aa, dd, iou_scores)
# IoU(bb, cc, iou_scores)
# IoU(bb, dd, iou_scores)
# IoU(cc, dd, iou_scores)

# Elapsed time: 0.05070000000007013 ms
# Elapsed time: 0.019599999999897477 ms
# Elapsed time: 0.018000000000295557 ms
# Elapsed time: 0.01749999999978158 ms
# Elapsed time: 0.01700000000015578 ms
# Elapsed time: 0.017999999999851468 ms
# --------------------------------------------------
# Elapsed time: 0.004000000000115023 ms
# Elapsed time: 0.0023000000002326715 ms
# Elapsed time: 0.0019999999998354667 ms
# Elapsed time: 0.0024999999999053557 ms
# Elapsed time: 0.0018999999999991246 ms
# Elapsed time: 0.002600000000185787 ms

#######################################################3plagiarism detection
