from collections import defaultdict, Counter
import math
import random
import copy
import numpy as np

tmp_delim = "_"

test_size, heldout_size = 20000, 40000

rnd_seeds = [42, 1337, 420, 228, 1488, 2023, 666, 451, 69, 404]
num_experiments = 10

stat_column = ["stat", "value"]


def read_txt_words(input_file, lang):
    enc = 'UTF-8' if lang == "EN" else 'ISO-8859-2'
    with open(input_file, 'r', encoding=enc) as input:
        text = input.read()
    word_list = text.lower().split("\n")[:-1]
    print(f"{input_file} - loaded words count:\t{len(word_list)}")
    return word_list


def data_split(input_file, output_file_prefix):
    enc = 'UTF-8' if output_file_prefix == "EN" else 'ISO-8859-2'
    with open(input_file, 'r', encoding=enc) as file:
        lines = file.readlines()

    lines = lines[:-1]

    print(f"{input_file} - Total words count:\t{len(lines)}")

    test_set = lines[-test_size:]
    held_out_set = lines[-(test_size + heldout_size):-test_size]
    train_set = lines[:-(test_size + heldout_size)]

    print(f"{input_file} - Test set words count:\t{len(test_set)}")
    print(f"{input_file} - Heldout set words count:\t{len(held_out_set)}")
    print(f"{input_file} - Train set words count:\t{len(train_set)}")

    with open(f"{output_file_prefix}_test.txt", 'w') as output_file:
        output_file.writelines(test_set)

    with open(f"{output_file_prefix}_heldout.txt", 'w') as output_file:
        output_file.writelines(held_out_set)

    with open(f"{output_file_prefix}_train.txt", 'w') as output_file:
        output_file.writelines(train_set)


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


def uniform_cond_prob(n_gram_counts):
    return 1 / n_gram_counts[0]


def unigram_cond_prob(w1, n_gram_counts):
    if (w1, ) not in n_gram_counts[1].keys():
        return uniform_cond_prob(n_gram_counts)
    return (n_gram_counts[1][(w1,)]) / n_gram_counts[0]


def bigram_cond_prob(w1, w2, n_gram_counts):
    if (w1, ) not in n_gram_counts[1].keys() or (w1, w2) not in n_gram_counts[2].keys():
        return uniform_cond_prob(n_gram_counts)
    return (n_gram_counts[2][(w1, w2)]) / (n_gram_counts[1][(w1,)])


def trigram_cond_prob(w1, w2, w3, n_gram_counts):
    if (w1, w2, w3) not in n_gram_counts[3].keys() or (w1, w2) not in n_gram_counts[2].keys():
        return uniform_cond_prob(n_gram_counts)
    return (n_gram_counts[3][(w1, w2, w3)]) / (n_gram_counts[2][(w1, w2)])


def model_prob(w1, w2, w3, n_gram_counts, lambdas=[1, 1, 1, 1]):
    return (lambdas[3] * trigram_cond_prob(w1, w2, w3, n_gram_counts) +
            lambdas[2] * bigram_cond_prob(w2, w3, n_gram_counts) +
            lambdas[1] * unigram_cond_prob(w3, n_gram_counts) +
            lambdas[0] * uniform_cond_prob(n_gram_counts))


def em_smooth_algo(seq, n_gram_counts, epsilon=1e-8, max_iterations=100):
    lambdas = np.array([0.25, 0.25, 0.25, 0.25])
    tri_pad_seq = pad_data_seq(seq, 2)

    print(f"EM smoothing for {len(tri_pad_seq)} words sequence")

    for iteration in range(max_iterations):

        expected_counts = np.zeros(4)
        for i in range(2, len(tri_pad_seq)):
            total_prob = model_prob(tri_pad_seq[i-2], tri_pad_seq[i-1], tri_pad_seq[i], n_gram_counts, lambdas)

            if total_prob > 0:
                expected_counts += np.array([
                    lambdas[0] * uniform_cond_prob(n_gram_counts) / total_prob,
                    lambdas[1] * unigram_cond_prob(tri_pad_seq[i], n_gram_counts) / total_prob,
                    lambdas[2] * bigram_cond_prob(tri_pad_seq[i-1], tri_pad_seq[i], n_gram_counts) / total_prob,
                    lambdas[3] * trigram_cond_prob(tri_pad_seq[i-2], tri_pad_seq[i-1], tri_pad_seq[i], n_gram_counts) / total_prob
                ])

        total_counts = np.sum(expected_counts)
        new_lambdas = expected_counts / total_counts

        new_lambdas /= sum(new_lambdas)

        convergence = np.abs(new_lambdas - lambdas) < epsilon

        print(f"{iteration} - {convergence} convergence")

        # print(iteration, expected_counts, new_lambdas, np.sum(new_lambdas), convergence)
        if np.all(convergence):
            break

        lambdas = new_lambdas

    return lambdas


def cross_entropy(seq, n_gram_counts, lambdas=[1, 1, 1, 1]):
    tri_pad_seq = pad_data_seq(seq, 2)

    print(f"Cross entropy for {len(tri_pad_seq)} words sequence")

    entropy = 0

    for i in range(2, len(tri_pad_seq)):
        total_prob = model_prob(tri_pad_seq[i - 2], tri_pad_seq[i - 1], tri_pad_seq[i], n_gram_counts, lambdas)
        entropy -= math.log2(total_prob)

    return entropy / len(tri_pad_seq)


def adjust_lambda(lambdas, percent, increase=True):
    ratio = percent / 100

    remain_total = lambdas[0] + lambdas[1] + lambdas[2]

    if increase:
        diff = (1 - lambdas[-1]) * ratio

        adjusted_lambdas = [
            lambdas[0] - diff * (lambdas[0] / remain_total),
            lambdas[1] - diff * (lambdas[1] / remain_total),
            lambdas[2] - diff * (lambdas[2] / remain_total),
            lambdas[3] + diff
        ]

    else:
        diff = lambdas[-1] * (1 - ratio)

        adjusted_lambdas = [
            lambdas[0] + diff * (lambdas[0] / remain_total),
            lambdas[1] + diff * (lambdas[1] / remain_total),
            lambdas[2] + diff * (lambdas[2] / remain_total),
            lambdas[3] * ratio
        ]

    adjusted_lambdas /= sum(adjusted_lambdas)

    # print(percent, np.sum(adjusted_lambdas), adjusted_lambdas, lambdas)

    return adjusted_lambdas



def comp_probs(words):
    word_pair_counts, word_counts = defaultdict(int), defaultdict(int)

    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i + 1]

        word_pair_counts[(current_word, next_word)] += 1
        word_counts[current_word] += 1

    # Compute probabilities P(i, j)
    word_pair_probabilities = {
        (word_i, word_j): count / len(words)
        for (word_i, word_j), count in word_pair_counts.items()
    }

    # Compute conditional probabilities P(j | i)
    conditional_probabilities = {
        (word_i, word_j): count / word_counts[word_i]
        for (word_i, word_j), count in word_pair_counts.items()
    }

    return word_pair_probabilities, conditional_probabilities


def comp_cond_entropy(word_pair_probabilities, conditional_probabilities):
    conditional_entropy = 0.0

    for (word_i, word_j), prob_ij in word_pair_probabilities.items():
        prob_j_given_i = conditional_probabilities.get((word_i, word_j), 0)
        if prob_j_given_i > 0:
            conditional_entropy += prob_ij * math.log2(prob_j_given_i)

    return -conditional_entropy


def comp_perplexity(conditional_entropy):
    return 2 ** int(conditional_entropy)


def mess_up_text_chars(text, mess_up_likelihood, seed):
    random.seed(seed)

    messed_up_text = list(copy.deepcopy(text))

    possible_chars = set(messed_up_text)
    possible_chars.remove(tmp_delim)

    total = len([char for char in messed_up_text if char != tmp_delim])
    mess_cnt = 0

    for i in range(len(messed_up_text)):
        if messed_up_text[i] == tmp_delim:
            continue

        if random.random() < mess_up_likelihood:
            mess_cnt += 1
            original_char = messed_up_text[i]

            chars = copy.deepcopy(possible_chars)
            chars.remove(original_char)
            new_char = random.choice(list(chars))

            messed_up_text[i] = new_char

    print(f"Changed {mess_cnt} characters in text of {total} characters - {round(mess_cnt/total, 2) * 100}% of text characters was messed up")
    return "".join(messed_up_text).split(tmp_delim)


def mess_up_text_words(words, mess_up_likelihood, seed):
    random.seed(seed)

    messed_up_words = copy.deepcopy(words)

    possible_words = set(messed_up_words)

    total = len(messed_up_words)
    mess_cnt = 0

    for i in range(len(messed_up_words)):
        if random.random() < mess_up_likelihood:
            mess_cnt += 1
            original_word = messed_up_words[i]

            w = copy.deepcopy(possible_words)
            w.remove(original_word)
            new_word = random.choice(list(w))

            messed_up_words[i] = new_word

    print(f"Changed {mess_cnt} words in text of {total} words - {round(mess_cnt/total, 2) * 100}% of text words was messed up")
    return messed_up_words


def run_char_experiment(text, likelihood):
    cond_entropies = []
    print(f"\tCharacter mess experiments for {likelihood} likelihood")
    for exp in range(num_experiments):
        seed = rnd_seeds[exp]
        mess_word_list = mess_up_text_chars(text, likelihood, seed)

        pair_probs, cond_probs = comp_probs(mess_word_list)
        entropy = comp_cond_entropy(pair_probs, cond_probs)
        cond_entropies.append(entropy)

        print(f"Experiment {exp + 1}\tSeed {seed}\tConditional entropy {entropy}")

    result = {
        "likelihood": likelihood,
        "min_entropy": min(cond_entropies),
        "max_entropy": max(cond_entropies),
        "avg_entropy": sum(cond_entropies) / num_experiments
    }

    return result


def run_word_experiment(words, likelihood):
    cond_entropies = []
    print(f"\tWord mess experiments for {likelihood} likelihood")
    for exp in range(num_experiments):
        seed = rnd_seeds[exp]
        mess_word_list = mess_up_text_words(words, likelihood, seed)

        pair_probs, cond_probs = comp_probs(mess_word_list)
        entropy = comp_cond_entropy(pair_probs, cond_probs)
        cond_entropies.append(entropy)

        print(f"Experiment {exp + 1}\tSeed {seed}\tConditional entropy {entropy}")

    result = {
        "likelihood": likelihood,
        "min_entropy": min(cond_entropies),
        "max_entropy": max(cond_entropies),
        "avg_entropy": sum(cond_entropies) / num_experiments
    }

    return result



