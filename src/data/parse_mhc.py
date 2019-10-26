# Parse the MHC Allele and Peptide Text File
from itertools import groupby

bdata_file = "bdata.20130222.mhci.txt"
alleles_file = "human_allele_seq.txt"


'''
Maps a given allele to a list of peptides that will bind with it
'''
def allele_to_peptide():
    allele_to_peptide = {}

    file = open(bdata_file)
    data = file.readlines()[1:]
    for line in data:
        parsed = line.split()
        allele = parsed[1]
        peptide_seq = parsed[3]

        if allele not in allele_to_peptide:
            allele_to_peptide[allele] = []
        allele_to_peptide[allele].append(peptide_seq)


    return allele_to_peptide

'''
Construct a mapping of allele to amino-acid sequence
@param: alleles_file: the file containing the FASTA alleles
'''
def allele_to_aminos():
    allele_to_amino = {}

    file = open(alleles_file)
    data = file.read()
    for pair in data.split("\n\n"):
        parsed = pair.split("\n")
        print(str(parsed))
        allele_name = parsed[0]
        sequence = ""
        for i in range(1, len(parsed)):
            sequence += parsed[i]

        allele_to_amino[allele_name] = sequence

    return allele_to_amino

'''
Generates a fasta file iterator
@param: alleles_file: the file containing the FASTA alleles
'''
def generate_fasta_iterator(alleles_file):
    fh = open(alleles_file)
    # ditch the boolean (x[0]) and just keep the header or sequence since
    # we know they alternate.
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

    for header in faiter:
        # drop the ">"
        headerStr = header.__next__()[1:].strip()

        # join all sequence lines to one.
        seq = "".join(s.strip() for s in faiter.__next__())

        yield (headerStr, seq)

'''
Generates the dataset we are looking for
(allele_seq, peptide_seq) --> binding affinity
@param: allele_to_seq_dict: a dictionary mapping of allele name to AA sequence
'''
def generate_dataset(allele_to_seq_dict):
    allele_to_sequence = {}

    file = open(bdata_file)
    data = file.readlines()[1:]
    for line in data:
        parsed = line.split()
        allele_name = parsed[1]
        if allele_name not in allele_to_seq_dict.keys():
            print("Allele not available: ", allele_name)
            continue
        allele_seq = allele_to_seq_dict[allele_name]
        peptide_seq = parsed[3]
        binding_affinity = parsed[5]

        key = (allele_seq, peptide_seq)
        allele_to_sequence[key] = binding_affinity

    return allele_to_sequence




a = allele_to_aminos()
b = allele_to_peptide()

intersection = set(a.keys()).intersection(set(b.keys()))
print("Intersection: ", intersection)

#allele_to_seq = allele_to_aminos(alleles_file)
#for a in allele_to_seq.keys():
#    print("Available allele: ", a)
#data = generate_dataset(allele_to_seq)
