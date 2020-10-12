# Parse the MHC Allele and Peptide Text File
import json
import matplotlib.pyplot as plt

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
Generates a ngram dataset
(allele_name, list of peptides that will bind)
'''
def generate_ngram_dataset(allele_to_peptides):
    dataset = {}

    for allele in allele_to_peptides:
        pairs = allele_to_peptides[allele]
        dataset[allele] = pairs
    return dataset




#allele_to_aminos = allele_to_aminos()
allele_to_peptides = allele_to_peptide()
# Question: How many peptides are associated with each allele? And are they unique? And how many alleles?
#num_alleles = 0
#num_peptides = []
#num_unique_peptides = []
#for a in allele_to_peptides.keys():
#    num_alleles += 1
#    val = allele_to_peptides[a]
#    peptides = []
#    for p, b in val:
#        peptides.append(p)

#    unique_peptides = set(peptides)
#    num_peptides.append(len(peptides))
#    num_unique_peptides.append(len(unique_peptides))

#print("Num alleles: ", num_alleles)
#print("Average number of non unique peptides: ", sum(num_peptides)/num_alleles)
#print("Average number of unique peptides: ", sum(num_unique_peptides)/num_alleles)
#print("Minimum number of peptides: ", min(num_peptides))
#print("Maximum number of peptides: ", max(num_peptides))
#num_bins = 100
#plt.hist(num_peptides, num_bins) #, normed=1)
#plt.show()


# This is the full dataset
#dataset = generate_dataset(allele_to_aminos, allele_to_peptides)
# Ngram dataset
ngram_dataset = generate_ngram_dataset(allele_to_peptides)
for allele in ngram_dataset:
    print(str(ngram_dataset[allele]))
    break

#with open("human_dataset.json", 'w') as f:
#    json.dump(dataset, f)

with open("ngram_dataset.json", 'w') as f:
    json.dump(ngram_dataset, f)



