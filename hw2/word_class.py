from bigram_table import *

class WCmodel:
    def __init__(self, bg_tab, ngram_counters):
        self.bg_tab = bg_tab

        self.min_class = 15

        self.min_occur = 10
        self.words_N = 8001

        self.ngram_counters = ngram_counters
        self.vocab = [w[0] for w in self.ngram_counters[1].keys()]
        self.popular = [w[0] for w, c in self.ngram_counters[1].items() if c >= self.min_occur]
        print(f"{len(self.vocab)} unique ( {len(self.popular)} frequent ) words found")

        print(f"{len(self.ngram_counters[2].values())} bigrams found")

        self.r = {}  #rk(w) V -> C mapping for most frequent
        for c, word in enumerate(self.popular):
            # print(c, word)
            self.r[word] = word
        self.class_N = len(self.r.keys())
        print(f"Initial word:class number:\t{len(self.r.keys())}")

        # self.bg_tab.get_class_counts(self.r)

        self.history = {}

        # self.bg_tab.get_mi_table(self.class_N)

        self.mi = self.bg_tab.get_MI_table(self.vocab, self.words_N)
        print(f"Total MI:\t{self.mi}")

        self.bg_tab.get_loss_table()

        self.H = {}  # merges history

    def algo(self, output):
        while self.class_N > self.min_class:
            records, pair = self.bg_tab.l(self.popular, self.words_N)
            self.H[self.class_N] = pair

            # print(pair, records[pair])
            print(f"{self.class_N}\t{pair}\t{records[pair]}")

            # c1, c2 = self.r[pair[0]], self.r[pair[1]]
            self.r, self.popular = self.bg_tab.merge(self.r, pair[0], pair[1], self.popular, self.words_N)
            print(self.class_N, len(list(set(self.r.values()))))

            print(f"Total MI:\t{sum(sum(self.bg_tab.df_MI.values))}")

            self.class_N -= 1
            self.words_N -= 1

        with open(output, "w") as file:
            for class_N, pair in self.H.items():
                pair_class = (self.r[pair[0]], self.r[pair[1]])
                file.write(f"{class_N}\t{pair}\t{pair_class}\n")

        return self.r

