import numpy as np
from utilities.notes_utils import multi_hot_encoding_12_tones

match_score = 1
mismatch_penalty = 0.3
rest_score = 0.05

def IoU(multihot1, multihot2):
    if sum(multihot1) == sum(multihot2) == 0:
        return rest_score
    return 2 * sum(multihot1 * multihot2) / (sum(multihot1) + sum(multihot2))

def local_alignment(A, B):
    scores = np.zeros((len(A)+1, len(B)+1))
    dir = np.zeros((len(A) + 1, len(B) + 1, 2))
    best = (0, (0, 0))
    for i in range(1, len(A)+1):
        for j in range(1, len(B)+1):
            iou = IoU(A[i-1], B[j-1])
            if iou > 0:
                scores[i, j] = scores[i-1, j-1] + iou * match_score
                dir[i, j] = [-1, -1]
            for dx, dy in ((-1, 0), (0, -1)):
                if scores[i+dx, j+dy] - mismatch_penalty > scores[i, j]:
                    scores[i, j] = scores[i+dx, j+dy] - mismatch_penalty
                    dir[i, j] = [dx, dy]
            if scores[i, j] > best[0]:
                best = (scores[i, j], (i, j))

    start = end = best[1]
    while not (dir[start] == [0, 0]).all():
        start = tuple(map(int, map(sum, zip(start, dir[start]))))

    return best[0], (start[0], end[0]), (start[1], end[1])


#                       [np.array([0, 0]), np.array([0, 1]), np.array([1, 1]), np.array([1, 0])]))

# print(local_alignment([np.array([0, 0]), np.array([0, 1]), np.array([1, 0])],

def find_plagiarism(song_path, database_path):
