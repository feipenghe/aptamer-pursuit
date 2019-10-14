# Parse the MHC Allele and Peptide Text File
from itertools import groupby

bdata_file = "/Users/aishwaryamandyam/Documents/Research/Masters MISL/aptamer-pursuit/bdata.20130222.mhci.txt"
alleles_file = "/Users/aishwaryamandyam/Documents/Research/Masters " \
                "MISL/aptamer-pursuit/alleles.txt"


'''
Explores the bdata_file
'''
def find_alleles():
    alleles = set()
    species = set()

    spec_to_prefix = {}

    file = open(bdata_file)
    data = file.readlines()[1:]
    for line in data:
        parsed = line.split()
        allele = parsed[1]
        spec = parsed[0]

        if spec not in spec_to_prefix:
            spec_to_prefix[spec] = []
        spec_to_prefix[spec].append(allele)

        species.add(spec)
        alleles.add(allele)

    print("Available Species: ", species)

    return spec_to_prefix

'''
Construct a mapping of allele to amino-acid sequence
@param: alleles_file: the file containing the FASTA alleles
'''
def allele_to_aminos(alleles_file):
    allele_to_amino = {}

    fasta_iterator = generate_fasta_iterator(alleles_file)
    for ff in fasta_iterator:
        header, seq = ff
        parsed_header = header.split(',')
        allele_name = parsed_header[1]
        #allele_to_amino[allele_name.replace(':', '')] = seq

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
            #print("Allele: ", allele_name)
            continue
        allele_seq = allele_to_seq_dict[allele_name]
        peptide_seq = parsed[3]
        binding_affinity = parsed[5]

        key = (allele_seq, peptide_seq)
        print("Key: ", key)
        print("Affinity: ", binding_affinity)
        allele_to_sequence[key] = binding_affinity

    return allele_to_sequence




dict = allele_to_aminos(alleles_file)
data = generate_dataset(dict)
