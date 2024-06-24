from word_class import *

input_file_cz = "texts/TEXTCZ1.txt"
input_file_en = "texts/TEXTEN1.txt"

input_file_tags_cz = "texts/TEXTCZ1.ptg"
input_file_tags_en = "texts/TEXTEN1.ptg"


def text_best_pmi(input_file, prefix):
    word_list = read_txt_words(input_file, prefix)

    counters = ngram_counters(word_list)

    pmi_stats = pmi_counters(counters)

    top20 = {k: pmi_stats[k] for k in list(pmi_stats)[:20]}

    output_file = f"tables/{prefix}_pmi_scores.txt"

    with open(output_file, "w") as f:
        for pair, pmi in top20.items():
            f.write(f"{pair}\t{pmi}\n")

    print(f"Top 20 word pairs PMI was saved to {output_file}")

    distant_pmi_stats = distant_pmi_counters(word_list, counters)

    distant_top20 = {k: distant_pmi_stats[k] for k in list(distant_pmi_stats)[:20]}

    output_file = f"tables/{prefix}_distant_pmi_scores.txt"

    with open(output_file, "w") as f:
        for pair, pmi in distant_top20.items():
            f.write(f"{pair}\t{pmi}\n")

    print(f"Top 20 distant word pairs PMI was saved to {output_file}")

    return top20, distant_top20


def text_class(input_file, prefix):
    class_N = 8000

    word_list, tag_list = read_ptg_words(input_file, prefix)
    test_seq = word_list[:class_N]

    cntr = ngram_counters(test_seq)
    bg_tab = BGtable(prefix)
    bg_tab.gen_bigram_table(cntr)

    word_class = WCmodel(bg_tab, cntr)

    mapping = word_class.algo(f"tables/{prefix}_history.txt")

    class_15 = {}
    for word, term in mapping.items():
        if term not in class_15.keys():
            class_15[term] = word
        else:
            class_15[term] += f"+{word}"

    with open(f"tables/{prefix}_15_class.txt", "w") as file:
        for term, words in class_15.items():
            file.write(f"term {term}\t: {words}\n")


# en_top20, en_dist_top20 = text_best_pmi(input_file_en, "EN")
# cz_top20, cz_dist_top20 = text_best_pmi(input_file_cz, "CZ")

text_class(input_file_tags_en, "EN")
text_class(input_file_tags_cz, "CZ")