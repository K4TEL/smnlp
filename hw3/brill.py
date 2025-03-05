import nltk
from nltk.tag import UnigramTagger, BigramTagger, brill, brill_trainer
from nltk.corpus.reader import TaggedCorpusReader
import random
import pickle
import os

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
    templates = [
        brill.Template(brill.Pos([-1])),  # Previous tag
        brill.Template(brill.Pos([1])),  # Next tag
        brill.Template(brill.Pos([-2])),  # Previous previous tag
        brill.Template(brill.Pos([2])),  # Next next tag
        brill.Template(brill.Pos([-2, -1])),  # Previous tags
        brill.Template(brill.Pos([1, 2])),  # Next tags
        brill.Template(brill.Word([-1])),  # Previous word
        brill.Template(brill.Word([1])),  # Next word
        brill.Template(brill.Word([-2])),  # Previous previous word
        brill.Template(brill.Word([2])),  # Next next word
        brill.Template(brill.Word([-2, -1])),  # Previous words
        brill.Template(brill.Word([1, 2])),  # Next words
    ]

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

# Main function to execute the process
def corpus_process(source_file: str, cz: bool = False):
    # Load and split the data
    data_T, data_H, data_S = load_and_split_data(source_file, cz)

    print(f'Train: {len(data_T)} words')
    print(f'Heldout: {len(data_H)} words')
    print(f'Test: {len(data_S)} words')

    # Prepare datasets
    train_sents, smooth_sents, test_sents = prepare_datasets(data_T, data_H, data_S)

    print(f'Train: {len(train_sents)} sentences')
    print(f'Heldout: {len(smooth_sents)} sentences')
    print(f'Test: {len(test_sents)} sentences')

    # Train the Brill tagger
    brill_tagger = train_brill_tagger(train_sents)

    # Evaluate the tagger
    accuracy = evaluate_tagger(brill_tagger, test_sents)
    print(f'Test Accuracy: {accuracy:.4f}')

    # Perform cross-validation
    accuracies, mean_accuracy, std_dev = cross_validation(data_T + data_H + data_S)
    print(f'Cross-average Accuracy: {mean_accuracy:.4f}')


if __name__ == "__main__":
    # Load data
    english_file = "TEXTEN2.ptg"
    czech_file = "TEXTCZ2.ptg"

    corpus_process(english_file)

    corpus_process(czech_file, True)

