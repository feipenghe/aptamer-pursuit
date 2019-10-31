# A bigram predicts the likelihood of the current element based on the previous element.
import json
import nltk
import random

# Construct a training dataset:
dataset_file = "../data/bigram_dataset.json"

with open(dataset_file, 'r') as f:
    bigram_dataset = json.load(f)

# Generate bigrams for each allele in the dataset
for allele in bigram_dataset:

    sentences = []
    proteins = bigram_dataset[allele]
    for p in proteins:
        sentences.append(p)

    # Generates the bigrams for this particular peptide set for this allele
    bigrams = {}
    for sentence in sentences:
        for i in range(len(sentence) - 2):
            seq = str(sentence[i:i+2])
            if seq not in bigrams.keys():
                bigrams[seq] = []
            bigrams[seq].append(sentence[i+2])

    # Generate the best 8 amino acid protein sequence
    # with a given tart letter
    start = sentences[0][0:2]
    output = start
    for i in range(6):
        if start not in bigrams.keys():
            break
        possible_chars = bigrams[start]
        next_char = possible_chars[random.randrange(len(possible_chars))]
        output += next_char
        start = output[len(output)-2:len(output)]

    print("Allele: " + str(allele) + ", Predicted Bigram Protein: " + str(output))





