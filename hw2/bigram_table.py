import copy
import pandas as pd
import os.path
from util import *

class BGtable:
    def __init__(self, prefix, counts=None):
        self.df_word_count = counts
        self.df_MI = None  # qk(l,r) MI
        self.bigram_N = None

        self.s = {}  # sk(a) row-column a MI sum

        self.df_loss = None

        self.w1_ll_w2 = None  # LL from left to list of right classes
        self.w2_ll_w1 = None  # LL from right to list of left classes

        self.L = {}
        self.prefix = prefix

    def gen_bigram_table(self, counters, output="tmp/bg.txt"):
        output = self.prefix + "_" + output
        if os.path.isfile(output):
            print(f"Loading table from {output}")
            self.df_word_count = pd.read_csv(output, sep="\t", index_col=0)

            first_ll_second, second_ll_first = get_ll(self.df_word_count)

            self.w1_ll_w2 = first_ll_second
            self.w2_ll_w1 = second_ll_first

            self.bigram_N = sum(sum(self.df_word_count.values))

            return self.df_word_count

        bi_cnt = counters[2]
        print(f"Computing bigram counts table for {len(bi_cnt.keys())} word pairs")

        first_words = list(set(bigram[0] for bigram in bi_cnt))
        second_words = list(set(bigram[1] for bigram in bi_cnt))

        first_ll_second, second_ll_first = {}, {}

        for first in first_words:
            first_ll_second[first] = collections.deque()

        for second in second_words:
            second_ll_first[second] = collections.deque()

        total_words = list(set(first_words + second_words))

        # print(len(first_words), len(second_words), len(total_words))

        df = pd.DataFrame(index=total_words, columns=total_words)

        for bigram, count in bi_cnt.items():
            df.at[bigram[0], bigram[1]] = count

            first_ll_second[bigram[0]].append(bigram[1])
            second_ll_first[bigram[1]].append(bigram[0])

        # print(df.info())

        df.fillna(0, inplace=True)

        self.df_word_count = df
        self.w1_ll_w2 = first_ll_second
        self.w2_ll_w1 = second_ll_first

        df.to_csv(output, sep="\t")
        print(f"Table saved to {output}")

        return df

    def get_MI_table(self, vocab, N, output="tmp/mi.txt"):
        output = self.prefix + "_" + output
        if os.path.isfile(output):
            print(f"Loading table from {output}")
            self.df_MI = pd.read_csv(output, sep="\t", index_col=0)
            return sum(sum(self.df_MI.values))
        else:
            return self.q_mi(vocab, N, output, save=True)


    def get_loss_table(self, output="tmp/loss.txt"):
        output = self.prefix + "_" + output
        if os.path.isfile(output):
            print(f"Loading table from {output}")
            self.df_loss = pd.read_csv(output, sep="\t", index_col=0)

    def q_mi(self, vocab, N, output="tmp/mi.txt", save=False):
        mi_total = 0

        df = pd.DataFrame(index=vocab, columns=vocab)

        print(f"Computing MI for vocabulary of {len(vocab)} words")
        for w1 in vocab:
            for w2 in vocab:
                mi = self.mi(w1, w2, N)
                df.loc[w1, w2] = mi
                mi_total += mi

        if save:
            df.to_csv(output, sep="\t")
        self.df_MI = df

        print(f"Total MI:\t{mi_total}")
        return mi_total

    def s_mi(self, popular, N):
        self.s = {}
        # print(vocab)
        print(f"Computing row-col MI for vocabulary of {len(popular)} words")
        for word in popular:
            self.s[word] = self.cross_term_mi(word, N)
        # print(f"{len(vocab)} words' row-col MI computed and saved")
        return self.s

    def bigram_count(self, left, right, word=True):
        df = self.df_word_count if word else self.df_class_count
        return df.loc[left, right]

    def marginal_count_L(self, row_word, word=True):
        df = self.df_word_count if word else self.df_class_count
        return df.loc[row_word, :].sum()

    def marginal_count_R(self, col_word, word=True):
        df = self.df_word_count if word else self.df_class_count
        return df.loc[:, col_word].sum()

    def mi(self, left, right, N, word=True, force=False):
        if self.df_MI is not None and not force:
            return self.df_MI.loc[left, right]

        pair_count = self.bigram_count(left, right, word)

        if pair_count == 0:
            return pair_count

        # print(N, self.marginal_count_L(left, word), self.marginal_count_R(right, word), pair_count)
        # print(item_mi(N, self.marginal_count_L(left, word), self.marginal_count_R(right, word), pair_count))

        return item_mi(N, self.marginal_count_L(left, word), self.marginal_count_R(right, word), pair_count)

    def mi_merged(self, left_list, right_list, N, word=True):
        pair_count = 0
        for merged_r in right_list:
            for merged_l in left_list:
                pair_count += self.bigram_count(merged_l, merged_r, word)
        if pair_count == 0:
            return pair_count

        marginal_L, marginal_R = 0, 0
        for merged_r in right_list:
            marginal_R += self.marginal_count_R(merged_r, word)
        for merged_l in left_list:
            marginal_L += self.marginal_count_L(merged_l, word)

        return item_mi(N, marginal_L, marginal_R, pair_count)

    def cross_term_mi(self, term, N, word=True):
        # print(self.w2_ll_w1)
        left_sum = sum([self.mi(left, term, N, word) for left in self.w2_ll_w1[term]])  # term column
        right_sum = sum([self.mi(term, right, N, word) for right in self.w1_ll_w2[term]])  # term row
        return left_sum + right_sum - self.mi(term, term, N, word)

    def substract(self, left_c, right_c, N, word=True):
        return self.s[left_c] + self.s[right_c] - \
            self.mi(left_c, right_c, N, word) - self.mi(right_c, left_c, N, word)

    def add(self, left_c, right_c, N, word=True):
        merged_c = [left_c, right_c]
        left_sum = sum([self.mi_merged([left], merged_c, N, word) for left in self.w2_ll_w1[left_c] + self.w2_ll_w1[right_c]])
        right_sum = sum([self.mi_merged(merged_c, [right], N, word) for right in self.w1_ll_w2[left_c] + self.w1_ll_w2[right_c]])

        return left_sum + right_sum + self.mi_merged(merged_c, merged_c, N, word)

    def loss(self, left_c, right_c, class_N, word=True):
        # print(self.mi(left_c, right_c, class_N, word), self.substract(left_c, right_c, class_N, word), self.merged_add(left_c, right_c, class_N))
        if self.df_loss is not None:
            return self.df_loss.loc[left_c, right_c]
        return self.substract(left_c, right_c, class_N, word) + self.add(left_c, right_c, class_N, word)

    def l(self, popular, N, output="tmp/loss.txt"):
        records = {}

        print(f"Computing loss for {len(popular)} words")

        # if self.df_loss is None:
        self.s_mi(popular, N)

        df_loss = pd.DataFrame(index=popular, columns=popular) if self.df_loss is None else self.df_loss

        # df_loss = self.df_loss if self.df_loss is not None else copy.deepcopy(self.df_word_count * 1.0)

        # print(df_loss)

        print(f"{len(popular)}/{N} terms iteration")

        for i, w1 in enumerate(popular):
            # print(f"picks for\t{w1}")
            for j in range(i+1, len(popular)):
                w2 = popular[j]

                pair_loss = self.loss(w1, w2, N)

                df_loss.loc[w1, w2] = pair_loss
                df_loss.loc[w2, w1] = pair_loss

                pair = (w1, w2)

                # print(pair_loss, pair)

                if w1 == w2:
                    continue

                records[pair] = pair_loss

        if self.df_loss is None:
            output = self.prefix + "_" + output
            df_loss.to_csv(output, sep="\t")
            self.df_loss = df_loss
        # print(df_loss)
        print(f"Loss for {len(self.df_loss.index)} terms was saved")

        best_pair = min(records, key=records.get)

        # print(df_loss)

        return records, best_pair

    def merge(self, r, parent, child, popular, N):
        print(f"Merging {child} to {parent}")

        for word, term in r.items():
            r[word] = parent if term == child else term
        popular.remove(child)

        print(f"Recomputing loss table")

        L = pd.DataFrame(index=popular, columns=popular)
        for w1 in popular:
            for w2 in popular:
                updated_loss = (self.df_loss.loc[w1, w2] - self.s[w1] - self.s[w2] +
                                self.mi_merged([w1, w2], [parent], N) + self.mi_merged([parent], [w1, w2], N) +
                                self.mi_merged([w1, w2], [child], N) + self.mi_merged([child], [w1, w2], N))
                L.loc[w1, w2] = updated_loss

        self.df_word_count.loc[:, parent] += self.df_word_count.loc[:, child]
        self.df_word_count.drop([child], axis=1, inplace=True)
        self.df_word_count.loc[parent, :] += self.df_word_count.loc[child, :]
        self.df_word_count.drop([child], axis=0, inplace=True)

        old_MI = copy.deepcopy(self.df_MI)

        to_check = (self.w1_ll_w2[parent] + self.w1_ll_w2[child] +
                    self.w2_ll_w1[parent] + self.w2_ll_w1[child])
        parent_L, parent_R = self.marginal_count_L(parent), self.marginal_count_R(parent)
        for word in to_check:
            if word == child:
                continue
            word_L, word_R = self.marginal_count_L(word), self.marginal_count_R(word)
            pair_WP = self.bigram_count(word, parent)
            if pair_WP > 0:
                self.df_MI.loc[word, parent] = item_mi(N-1, word_L, parent_R, pair_WP)

            pair_PW = self.bigram_count(parent, word)
            if pair_PW > 0:
                self.df_MI.loc[parent, word] = item_mi(N - 1, parent_L, word_R, pair_PW)
        self.df_MI.drop(index=child, inplace=True)
        self.df_MI.drop(columns=child, inplace=True)

        for first, second_list in self.w1_ll_w2.items():
            if child in second_list:
                self.w1_ll_w2[first].remove(child)
                if parent not in second_list:
                    self.w1_ll_w2[first].append(parent)
            if first == child:
                left_second_list = self.w1_ll_w2[parent]
                union = set(left_second_list + second_list)
                self.w1_ll_w2[parent] = collections.deque(union)
        self.w1_ll_w2.pop(child, None)

        for second, first_list in self.w2_ll_w1.items():
            if child in first_list:
                self.w2_ll_w1[second].remove(child)
                if parent not in first_list:
                    self.w2_ll_w1[second].append(parent)
            if second == child:
                left_first_list = self.w2_ll_w1[parent]
                union = set(left_first_list + first_list)
                self.w2_ll_w1[parent] = collections.deque(union)
        self.w2_ll_w1.pop(child, None)

        s = {}
        for term, cross_mi in self.s.items():
            if term == child:
                continue
            s[term] = (self.s[term] - old_MI.loc[term, parent] - old_MI.loc[term, child]
                            - old_MI.loc[parent, term] - old_MI.loc[child, term] +
                            self.mi(parent, term, N-1) + self.mi(term, parent, N-1))
        self.s = s

        for w1 in popular:
            for w2 in popular:
                loss = L.loc[w1, w2]
                final_loss = (loss + self.s[w1] + self.s[w2] +
                             self.mi_merged([w1, w2], [parent], N-1) +
                             self.mi_merged([parent], [w1, w2], N-1))
                self.df_loss.loc[w1, w2] = final_loss
                self.df_loss.loc[w2, w1] = final_loss
        self.df_loss.drop(index=child, inplace=True)
        self.df_loss.drop(columns=child, inplace=True)

        return r, popular