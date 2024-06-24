import collections
from collections import Counter
import math


def read_txt_words(input_file, lang):
    enc = 'UTF-8' if lang == "EN" else 'ISO-8859-2'
    with open(input_file, 'r', encoding=enc) as input:
        text = input.read()
    word_list = text.lower().split("\n")[:-1]
    print(f"{input_file} - loaded words count:\t{len(word_list)}")
    return word_list


def read_ptg_words(input_file, lang):
    enc = 'UTF-8' if lang == "EN" else 'ISO-8859-2'
    with open(input_file, 'r', encoding=enc) as input:
        text = input.read()
    line_list = text.lower().split("\n")[:-1]
    word_list = [line.split("/")[0] for line in line_list]
    tag_list = [line.split("/")[1] for line in line_list]
    print(f"{input_file} - loaded words count:\t{len(word_list)}")
    return word_list, tag_list


def pad_data_seq(seq, n=1, padding="<s>"):
    return [padding * n] + seq + [padding * n]


def ngram_counters(seq):
    uni_cnt, bi_cnt, tri_cnt = Counter(), Counter(), Counter()

    bi_pad_seq, tri_pad_seq = pad_data_seq(seq), pad_data_seq(seq, 2)

    uni_cnt.update([(gram,) for gram in bi_pad_seq])
    bi_cnt.update([(gram1, gram2) for gram1, gram2 in zip(bi_pad_seq, bi_pad_seq[1:])])
    tri_cnt.update([(gram1, gram2, gram3) for gram1, gram2, gram3 in zip(tri_pad_seq, tri_pad_seq[1:], tri_pad_seq[2:])])

    uf_cnt = len(set(bi_pad_seq))

    n_gram_counts = {
        3: tri_cnt,
        2: bi_cnt,
        1: uni_cnt,
        0: uf_cnt
    }

    print(f"n-gram model of {len(bi_pad_seq)} words sequence:")
    print(f"3-grams:\t{len(tri_cnt.keys())}")
    print(f"2-grams:\t{len(bi_cnt.keys())}")
    print(f"1-grams:\t{len(uni_cnt.keys())}")

    return n_gram_counts

def pmi_counters(counters):
    pairs = counters[2]

    result = {}

    for pair, cnt in pairs.items():
        w1, w2 = pair[0], pair[1]

        c1 = counters[1][(w1,)]
        c2 = counters[1][(w2,)]

        if c1 > 9 and c2 > 9:
            pmi = math.log2((cnt * counters[0]) / (c1 * c2))

            result[pair] = pmi

    res_sorted = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    return res_sorted


def distant_pmi_counters(seq, counters):
    result = {}

    for i, word in enumerate(seq):
        if counters[1][(word,)] < 10:
            continue

        distant_neighbors = seq[i-51:i-2] + seq[i+2:i+51]
        popular_dist = [w for w in distant_neighbors if counters[1][(w,)] > 9]
        exist_dist_pair = [w for w in popular_dist if counters[2][(word, w)] > 0]

        for dist_word in exist_dist_pair:
            if (word, dist_word) in result.keys():
                continue

            joint = counters[2][(word, dist_word)]

            c1 = counters[1][(word,)]
            c2 = counters[1][(dist_word,)]

            pmi = math.log2((joint * counters[0]) / (c1 * c2))

            result[(word, dist_word)] = pmi

    res_sorted = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    return res_sorted


def get_ll(df):
    first_list, second_list = list(df.index), list(df.columns)

    first_ll_second, second_ll_first = {}, {}

    def nonzero_columns(row):
        return list(row.index[row != 0])

    nonzero_columns_per_row = df.apply(nonzero_columns, axis=1)
    nonzero_rows_per_column = df.apply(nonzero_columns, axis=0)

    for first in first_list:
        first_ll_second[first] = collections.deque(nonzero_columns_per_row[first])
    for second in second_list:
        second_ll_first[second] = collections.deque(nonzero_rows_per_column[second])

    return first_ll_second, second_ll_first


def item_mi(total, c1, c2, pair):
    if (pair * total) / (c1 * c2) > 0:
        return pair * math.log2((pair * total) / (c1 * c2)) / total
    else:
        return 0


