import nltk
from nltk.tag import UnigramTagger, BigramTagger, brill, brill_trainer
from nltk.corpus.reader import TaggedCorpusReader
import random
import pickle
import os
import csv

# Function to load and split the data
def load_and_split_data(file_path: str, cz: bool = False) -> (list, list, list):
    # Load the corpus
    codec = 'ISO-8859-2' if cz else 'UTF-8'
    reader = TaggedCorpusReader('.', file_path, sep='/', encoding=codec)
    sentences = list(reader.tagged_sents())

    # Flatten the list of sentences into a list of words
    words = [word for sentence in sentences for word in sentence]

    # Define the sizes for testing (S), smoothing (H), and training (T) datasets
    test_size = 40000
    smooth_size = 20000

    # Split the data
    data_S = words[-test_size:]
    data_H = words[-(test_size + smooth_size):-test_size]
    data_T = words[:-(test_size + smooth_size)]

    return data_T, data_H, data_S

# Function to prepare datasets for training and evaluation
def prepare_datasets(data_T: list[str], data_H: list[str], data_S: list[str]) -> (list, list, list):
    # Convert flat word lists back into sentences
    def to_sentences(word_list: list[str]) -> list[list[str]]:
        sentences = []
        sentence = []
        for word in word_list:
            if word[0] == '.':
                sentence.append(word)
                sentences.append(sentence)
                sentence = []
            else:
                sentence.append(word)
        if sentence:
            sentences.append(sentence)
        return sentences

    train_sents = to_sentences(data_T)
    smooth_sents = to_sentences(data_H)
    test_sents = to_sentences(data_S)

    return train_sents, smooth_sents, test_sents

# Function to train the Brill tagger
def train_brill_tagger(train_sents: list[list[str]], max_rules: int = 200, min_score: int = 2) -> brill.BrillTagger:
    # Define the baseline tagger
    baseline_tagger = BigramTagger(train_sents, backoff=UnigramTagger(train_sents))

    # Define Brill's templates
    templates = []
    for span in range(-2, 3):
        templates.append(brill.Template(brill.Word([span])))

        span_end = span + 1
        while span_end < 3:
            templates.append(brill.Template(brill.Word([span, span_end])))

            if span < 1:
                templates.append(brill.Template(brill.Pos([span]), brill.Word([span, span_end])))

                if span_end < 1:
                    templates.append(brill.Template(brill.Pos([span, span_end]), brill.Word([span, span_end])))

            if span_end-span < 3:
                extra_pos = span - 2 if span > -1 else span_end + 2
                if extra_pos > 0 and span_end > 0:
                    pass
                else:  # only -2 or 2
                    templates.append(brill.Template(brill.Word([span, span_end]), brill.Word([extra_pos])))

                    if extra_pos < 0:
                        templates.append(brill.Template(brill.Word([span, span_end]), brill.Word([extra_pos]), brill.Pos([extra_pos])))

            span_end += 1

        if span < 1:
            templates.append(brill.Template(brill.Pos([span])))
            templates.append(brill.Template(brill.Pos([span]), brill.Word([span])))

        span_end = span + 1
        while span_end < 1:
            templates.append(brill.Template(brill.Pos([span, span_end])))
            span_end += 1

        if span == -2:
            templates.append(brill.Template(brill.Pos([span]), brill.Pos([0])))
            templates.append(brill.Template(brill.Pos([span]), brill.Pos([0]), brill.Word([span])))

            span_end = span + 1
            while span_end < 3:
                templates.append(brill.Template(brill.Pos([span]), brill.Pos([0]), brill.Word([span, span_end])))
                span_end += 1

    # Initialize the Brill tagger trainer
    trainer = brill_trainer.BrillTaggerTrainer(baseline_tagger, templates, trace=2)

    # Train the Brill tagger
    brill_tagger = trainer.train(train_sents, max_rules=max_rules, min_score=min_score)

    return brill_tagger

# Function to evaluate the tagger
def evaluate_tagger(tagger: brill.BrillTagger, test_sents: list[list[str]]) -> float:
    accuracy = tagger.evaluate(test_sents)
    return accuracy

# Function to perform cross-validation
def cross_validation(data, k: int = 5) -> (list, float, float):
    random.shuffle(data)
    fold_size = len(data) // k
    accuracies = []

    for i in range(k):
        print(f'Fold {i + 1}/{k}')
        test_data = data[i * fold_size:(i + 1) * fold_size]
        train_data = data[:i * fold_size] + data[(i + 1) * fold_size:]

        train_sents = prepare_datasets(train_data, [], [])[0]
        test_sents = prepare_datasets([], [], test_data)[2]

        tagger = train_brill_tagger(train_sents)
        accuracy = evaluate_tagger(tagger, test_sents)
        accuracies.append(accuracy)

    mean_accuracy = sum(accuracies) / k
    std_dev = (sum([(x - mean_accuracy) ** 2 for x in accuracies]) / k) ** 0.5

    return accuracies, mean_accuracy, std_dev

# Function to save the trained tagger
def save_tagger(tagger: brill.BrillTagger, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(tagger, f)

# Function to load a trained tagger
def load_tagger(file_name: str) -> brill.BrillTagger:
    with open(file_name, 'rb') as f:
        tagger = pickle.load(f)
    return tagger

# Function to write results to CSV
def write_results_to_csv(results: list[dict], file_name: str):
    with open(file_name, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['LANG', 'STD', 'ACC'])
        writer.writeheader()
        writer.writerows(results)

# Main function to execute the process
def corpus_process(source_file: str, language: str, cz: bool = False):
    results = []

    data_T, data_H, data_S = load_and_split_data(source_file, cz)

    train_sents, smooth_sents, test_sents = prepare_datasets(data_T, data_H, data_S)

    # Train the Brill tagger
    brill_tagger = train_brill_tagger(train_sents)

    tagger_rules = brill_tagger.rules()

    with open(f"{language}_rules.txt", "w") as f:
        for rule in tagger_rules:
            f.write(f"{rule}\n")

    # Evaluate the tagger on test data
    test_accuracy = evaluate_tagger(brill_tagger, test_sents)
    results.append({'LANG': language, 'STD': 0, 'ACC': test_accuracy})

    # Perform cross-validation
    cross_val_accuracies, mean_acc, mean_std = cross_validation(data_T + data_H + data_S)

    results.append({'LANG': language, 'STD': mean_std, 'ACC': mean_acc})

    for i, accuracy in enumerate(cross_val_accuracies, start=1):
        print(f"Cross-validation fold {i}: {accuracy}")
        # results.append({'LANG': language, 'ITER': i, 'ACC': accuracy})

    return results

if __name__ == "__main__":
    english_file = "TEXTEN2.ptg"
    czech_file = "TEXTCZ2.ptg"

    all_results = []

    # Process English corpus
    english_results = corpus_process(english_file, "EN")
    all_results.extend(english_results)

    # Process Czech corpus
    czech_results = corpus_process(czech_file, "CZ", cz=True)
    all_results.extend(czech_results)

    # Write results to CSV
    write_results_to_csv(all_results, "brill_results.csv")

