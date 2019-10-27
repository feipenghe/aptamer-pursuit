# A bigram predicts the likelihood of the current element based on the previous element.
import json
import nltk

# Construct a training dataset:
dataset_file = "../data/bigram_dataset.json"

with open(dataset_file, 'r') as f:
    bigram_dataset = json.load(f)

# Generate bigrams for each allele in the dataset
for allele in bigram_dataset:
    sentences = []
    proteins = bigram_dataset[allele]
    for p in proteins:
        sentences.append(list(p))

    # Generates the bigrams for this particular peptide set for this allele
    bigram_list = []
    for sentence in sentences:
        bigram_list.extend(list(nltk.bigrams(sentence)))

    print(sentences)
    break


# Generate using the first couple of words from dataset (word is a protein)
# https://stackabuse.com/python-for-nlp-developing-an-automatic-text-filler-using-n-grams/ --> this is a useful link to understand


