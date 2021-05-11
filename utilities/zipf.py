import numpy as np
import matplotlib.pyplot as plt

from utilities.analyze import rmse


def generate_words(num, size_limit):
    s = {}
    ans = {}
    tot = 0
    rnds = np.random.randint(0, 24, num)
    for i in range(num):
        #print(rnds[i])
        if rnds[i] == 0:
            if len(s) == 0: continue
            if frozenset(s) in ans:
                ans[frozenset(s)] += 1
            else:
                print(frozenset(s))
                ans[frozenset(s)] = 1
            tot += 1
            s = {}
        else:
            if len(s) == 0: s = {rnds[i]}
            else: s.add(rnds[i])
        if len(s) > size_limit:
            s = {}

    y = np.array([ans[a] for a in ans])
    y[::-1].sort()
    offset = 1
    x = 1 + np.array(list(range(len(y) - offset)), dtype=np.float)
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(x, y[offset:])
    print(y)
    plt.show()


def generate_words_2(num, size_limit):
    s = {}
    ans = {}
    tot = 0
    alphabet_size = 13
    for i in range(num):
        #print(rnds[i])
        rnd = np.random.randint(0, alphabet_size + 1 - len(s))
        if rnd == 0:
            if len(s) == 0: continue
            # Do not add sets with more than size_limit elements.
            if len(s) > size_limit:
                s = {}
                continue
            if frozenset(s) in ans:
                ans[frozenset(s)] += 1
            else:
                print(frozenset(s))
                ans[frozenset(s)] = 1
            tot += 1
            s = {}
        else:
            idx = 0
            val = -1
            # Find the next unused character.
            # TODO: Maybe choose the next character
            # with probability proportional to Zipf's distribution.
            for i in range(1, alphabet_size + 1):
                if i not in s:
                    idx += 1
                if idx == rnd:
                    val = i
                    break
            if len(s) == 0: s = {val}
            else: s.add(val)

    y = np.array([ans[a] for a in ans])
    y[::-1].sort()
    offset = 1
    x = 1 + np.array(list(range(len(y) - offset)), dtype=np.float)
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(x, y[offset:])
    # print(y)
    plt.show()


def process_text(limit):
    with open("..\\assets\\shakespeare.txt", "r") as myfile:
        data = myfile.read().translate(str.maketrans(',.?-;{}}[]()/:!', '               ')).translate(str.maketrans('\n0123456789\'"', '%%%%%%%%%%%%%')).replace('%', ' ').lower()
    words = data.split(' ')
    word_sets = [frozenset(word) for word in words]
    cnt = {}
    for ws in word_sets:
        if len(ws) > limit: continue
        if ws not in cnt:
            print(ws)
            cnt[ws] = 1
        else:
            cnt[ws] += 1
    y = np.array([cnt[a] for a in cnt])
    y[::-1].sort()
    offset = 1
    x = 1 + np.array(list(range(len(y) - offset)), dtype=np.float)
    plt.xscale('log')
    plt.yscale('log')
    lx = np.log10(x)
    ly = np.log10(y[offset:])
    m, b = np.polyfit(lx, ly, 1)
    rmse_ = rmse(m*lx + b, ly)
    plt.plot(10**lx, 10**(m*lx+b))
    plt.scatter(x, y[offset:])
    plt.title(f"Zipf's law for Sh, RMSE: {rmse_}")
    print(y)
    plt.show()


# Model that generates each word by randomly sampling
# one that does not exist in the set.
# generate_words_2(50000, size_limit=4)
# generate_words_2(500000, size_limit=4)
# generate_words_2(5000000, size_limit=4)
# generate_words_2(5000000, size_limit=100)

# It seems that text grouping by character set
# generates a Zipfian distribution (at least for Alice and Shakespear).
process_text(limit=4)

# A possible weaker model.
# generate_words(500000)
