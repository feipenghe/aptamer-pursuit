# A bigram predicts the likelihood of the current element based on the previous element.
import json
import nltk
import random

# Construct a training dataset:
dataset_file = "../data/ngram_dataset.json"

with open(dataset_file, 'r') as f:
    ngram_dataset = json.load(f)

n = 3

# Generate predicted protein for each allele in the dataset separately
predicted_proteins = []
for allele in ngram_dataset:

    sentences = []
    proteins = ngram_dataset[allele]
    for p in proteins:
        sentences.append(p)

    # Generates the bigrams for this particular peptide set for this allele
    ngrams = {}
    for sentence in sentences:
        for i in range(len(sentence) - n):
            seq = str(sentence[i:i+n])
            if seq not in ngrams.keys():
                ngrams[seq] = []
            ngrams[seq].append(sentence[i+n])

    # Generate the best 8 amino acid protein sequence
    # with a given start letters
    start = sentences[0][0:n]
    output = start
    for i in range(6):
        if start not in ngrams.keys():
            break
        possible_chars = ngrams[start]
        next_char = possible_chars[random.randrange(len(possible_chars))]
        output += next_char
        start = output[len(output)-n:len(output)]
    predicted_proteins.append(output)
    print("Allele: " + str(allele) + ", Predicted Ngram Protein: " + str(output))

def generate_stats(predicted_proteins):
    # Shortest Length
    shortest_length = 8
    for i in range(len(predicted_proteins)):
        shortest_length = min(shortest_length, len(predicted_proteins[i]))
    # Longest Length = 8

    print("Shortest Length Peptide: ", shortest_length)

generate_stats(predicted_proteins)





