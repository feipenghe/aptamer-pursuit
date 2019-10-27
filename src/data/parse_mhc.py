# Parse the MHC Allele and Peptide Text File
import json

bdata_file = "bdata.20130222.mhci.txt"
alleles_file = "human_allele_seq.txt"


'''
Maps a given allele to a list of (peptide, binding affinity)
'''
def allele_to_peptide():
    allele_to_peptide = {}

    file = open(bdata_file)
    data = file.readlines()[1:]
    for line in data:
        parsed = line.split()
        allele = parsed[1]
        peptide_seq = parsed[3]
        binding_affinity = parsed[5]

        if allele not in allele_to_peptide:
            allele_to_peptide[allele] = []
        allele_to_peptide[allele].append((peptide_seq, binding_affinity))


    return allele_to_peptide

'''
Construct a mapping of allele to amino-acid sequence
'''
def allele_to_aminos():
    allele_to_amino = {}

    file = open(alleles_file)
    data = file.read()
    for pair in data.split("\n\n"):
        parsed = pair.split("\n")
        allele_name = parsed[0]
        sequence = ""
        for i in range(1, len(parsed)):
            sequence += parsed[i]

        allele_to_amino[allele_name] = sequence

    return allele_to_amino

'''
Generates the dataset we are looking for
(allele_seq, peptide_seq) --> binding affinity
@param: allele_to_seq_dict: a dictionary mapping of allele name to AA sequence
'''
def generate_dataset(allele_to_aminos, allele_to_peptides):
    dataset = {}

    for allele in allele_to_peptides:
        if allele not in allele_to_aminos:
            continue
        aa_seq = allele_to_aminos[allele]
        for p, b in allele_to_peptides[allele]:
            key = str(aa_seq) + "," + str(p)
            dataset[key] = b
    return dataset

'''
Generates a bigram dataset
(allele_name, list of peptides that will bind)
'''
def generate_bigram_dataset(allele_to_aminos, allele_to_peptides):
    dataset = {}

    for allele in allele_to_peptides:
        pairs = allele_to_peptides[allele]
        peptides = []
        for p, b in pairs:
            peptides.append(p)

        dataset[allele] = peptides
    return dataset




allele_to_aminos = allele_to_aminos()
allele_to_peptides = allele_to_peptide()
# This is the full dataset
dataset = generate_dataset(allele_to_aminos, allele_to_peptides)
# Bigram dataset
bigram_dataset = generate_bigram_dataset(allele_to_aminos, allele_to_peptides)

with open("human_dataset.json", 'w') as f:
    json.dump(dataset, f)

with open("bigram_dataset.json", 'w') as f:
    json.dump(bigram_dataset, f)



