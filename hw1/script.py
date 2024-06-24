import pandas as pd
from util import *

input_file_cz = "texts/TEXTCZ1.txt"
input_file_en = "texts/TEXTEN1.txt"

likelihoods = [0.1, 0.05, 0.01, 0.001, 0.0001]


def text_stats(words):
    df = pd.DataFrame(columns=stat_column)

    words_num = len(words)
    df.loc[len(df.index)] = ['words', words_num]

    chars_total_num = len("".join(words))
    df.loc[len(df.index)] = ['chars_total', chars_total_num]

    df.loc[len(df.index)] = ['chars_word', chars_total_num / words_num]

    word_counts = Counter(words)
    most_frequent_word, most_frequent_count = word_counts.most_common(1)[0]
    df.loc[len(df.index)] = ['highest_word_freq', most_frequent_count]

    df.loc[len(df.index)] = ['rare_words_count', sum(1 for count in word_counts.values() if count == 1)]

    pair_probabilities, conditional_probabilities = comp_probs(words)
    conditional_entropy = comp_cond_entropy(pair_probabilities, conditional_probabilities)

    df.loc[len(df.index)] = ['cond_entropy', conditional_entropy]

    df.loc[len(df.index)] = ['perplexity', comp_perplexity(conditional_entropy)]

    return df


def txt_entropy(input_file, output_file_pref):
    word_list = read_txt_words(input_file, output_file_pref)
    text = tmp_delim.join(word_list)

    stats_df = text_stats(word_list)
    stats_df.to_csv(f"{output_file_pref}_stats.txt", sep="\t", index=False)
    print(stats_df)

    char_mess_results = []
    for like in likelihoods:
        measures = run_char_experiment(text, like)
        char_mess_results.append(measures)

    char_mess_df = pd.DataFrame(char_mess_results)
    print(char_mess_df)

    char_mess_df.to_csv(f"{output_file_pref}_char_mess.txt", sep="\t", index=False)

    word_mess_results = []
    for like in likelihoods:
        measures = run_word_experiment(word_list, like)
        word_mess_results.append(measures)

    word_mess_df = pd.DataFrame(word_mess_results)
    word_mess_df.to_csv(f"{output_file_pref}_word_mess.txt", sep="\t", index=False)
    print(word_mess_df)

    return stats_df, char_mess_df, word_mess_df


def txt_model_cross_entropy(input_file_pref):
    word_list_train = read_txt_words(f"{input_file_pref}_train.txt", input_file_pref)
    word_list_heldout = read_txt_words(f"{input_file_pref}_heldout.txt", input_file_pref)
    word_list_test = read_txt_words(f"{input_file_pref}_test.txt", input_file_pref)

    n_gram_counts = ngram_counters(word_list_train)

    lambdas = em_smooth_algo(word_list_heldout, n_gram_counts)
    print("Optimal Lambdas:", lambdas)

    adj_results = [{
        "adjust": "None",
        "percent": 0,
        "cross_entropy": cross_entropy(word_list_test, n_gram_counts, lambdas)
    }]

    print(adj_results[0]["cross_entropy"])

    for percent in range(0, 100, 10):
        adj_lambdas = adjust_lambda(lambdas, percent, False)

        res = {
            "adjust": "decrease",
            "percent": percent,
            "cross_entropy": cross_entropy(word_list_test, n_gram_counts, adj_lambdas)
        }

        print(f"Decrease by {percent}:\t{res['cross_entropy']}")

        adj_results.append(res)

    percentage_range = list(range(10, 100, 10)) + [95, 99]
    for percent in percentage_range:
        adj_lambdas = adjust_lambda(lambdas, percent)

        res = {
            "adjust": "increase",
            "percent": percent,
            "cross_entropy": cross_entropy(word_list_test, n_gram_counts, adj_lambdas)
        }

        print(f"Increase by {percent}:\t{res['cross_entropy']}")

        adj_results.append(res)

    adj_stats_df = pd.DataFrame(adj_results)
    adj_stats_df.to_csv(f"{input_file_pref}_adj_lambda_stats.txt", sep="\t", index=False)

    print(adj_stats_df)

    return lambdas, adj_stats_df


stats_df_CZ, char_mess_df_CZ, word_mess_df_CZ = txt_entropy(input_file_cz, "CZ")
stats_df_EN, char_mess_df_EN, word_mess_df_EN = txt_entropy(input_file_cz, "EN")

data_split(input_file_en, "EN")
lambdas_EN, adj_stats_df_EN = txt_model_cross_entropy("EN")

data_split(input_file_cz, "CZ")
lambdas_CZ, adj_stats_df_CZ = txt_model_cross_entropy("CZ")


# stats_df_CZ = pd.read_csv("CZ_stats.txt", sep="\t")
# stats_df_EN = pd.read_csv("EN_stats.txt", sep="\t")
# df = pd.merge(stats_df_EN, stats_df_CZ, right_on="stat", left_on="stat", how="outer")
# df.rename(columns={"value_x": "EN", "value_y": "CZ"}, inplace=True)
# print(df)
# df.to_csv("stats.txt", sep="\t", index=False)
