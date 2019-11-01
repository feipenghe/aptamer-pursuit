# A bigram predicts the likelihood of the current element based on the previous element.
import json
import nltk
import random

# Construct a training dataset:
dataset_file = "../data/ngram_dataset.json"

with open(dataset_file, 'r') as f:
    ngram_dataset = json.load(f)


'''
Generate predicted protein for each allele in the dataset separately
@param: n = the number of characters to use in predictions. (i.e. bigram means n=2)
@param: seq_length = the length of peptide sequence to generate. 
'''
def predict_proteins(n=3, seq_length=8):
    predicted_proteins = []
    for allele in ngram_dataset:

        sentences = []
        proteins = ngram_dataset[allele]
        for p in proteins:
            sentences.append(p)

        # Generates the ngrams for this particular peptide set for this allele, with n value
        ngrams = {}
        for sentence in sentences:
            for i in range(len(sentence) - n):
                seq = str(sentence[i:i+n])
                if seq not in ngrams.keys():
                    ngrams[seq] = []
                ngrams[seq].append(sentence[i+n])

        # Generate a "seq_length" amino acid protein sequence
        # with a given start letters

        # For now, I'm just using the start of the first peptide sequence that sticks to this allele
        # TODO: get the peptide sequence that binds the best to this allele
        start = sentences[0][0:n]
        output = start
        for i in range(seq_length - n):
            if start not in ngrams.keys():
                break
            possible_chars = ngrams[start]
            next_char = possible_chars[random.randrange(len(possible_chars))]
            output += next_char
            start = output[len(output)-n:len(output)]
        predicted_proteins.append(output)
        print("Allele: " + str(allele) + ", Predicted Protein: " + str(output))
    return predicted_proteins

'''
Generate statistics about the proteins that I predicted.
TODO: Understand motifs, right now it just looks at length
@param: predicted_proteins = a list of predicted proteins (for all alleles)
@param: seq_length = the length of peptide sequence to generate. 
'''
def generate_stats(predicted_proteins, seq_length):
    # Shortest Length
    shortest_length = seq_length
    longest_length = 0
    for i in range(len(predicted_proteins)):
        shortest_length = min(shortest_length, len(predicted_proteins[i]))
        longest_length = max(longest_length, len(predicted_proteins[i]))

    print("-----Stats-----")
    print("Shortest Length Peptide: ", shortest_length)
    print("Longest Length Peptide: ", longest_length)


predicted_proteins = predict_proteins(n=1, seq_length=8)
generate_stats(predicted_proteins, seq_length=8)





