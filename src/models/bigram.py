# A bigram predicts the likelihood of the current element based on the previous element.
import json

# Construct a training dataset:
dataset_file = "../data/bigram_dataset.json"

with open(dataset_file, 'r') as f:
    bigram_dataset = json.load(f)
    for allele in bigram_dataset:
        print("Allele: ", allele)

# Generate bigrams for each allele in the dataset



# Write the model


# Train the model


# Test the model


