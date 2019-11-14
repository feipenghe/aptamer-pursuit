import os, sys
import json
import csv

pseudosequences = "class1_pseudosequences.csv"
bdata_file = "bdata.20130222.mhci.txt"

# Allele name to the AA pseudosequence
name_to_sequence = {}
with open(pseudosequences) as f:
	read_csv = csv.reader(f, delimiter=',')
	for row in read_csv:
		allele = row[0]
		sequence = row[1]
		name_to_sequence[allele] = sequence

file = open(bdata_file)
bdata_lines = file.readlines()[1:]

in_file = 0
not_in_file = 0

comprehensive_dataset = {}
for line in bdata_lines:
	parsed = line.split()
	allele = parsed[1]
	peptide_seq = parsed[3]
	binding_affinity = parsed[5]

	xallele = allele.replace('*', '')
	if xallele in name_to_sequence:
		in_file += 1
		aa_seq = name_to_sequence[xallele]

		if aa_seq not in comprehensive_dataset.keys():
			comprehensive_dataset[aa_seq] = []
		comprehensive_dataset[aa_seq].append((peptide_seq, binding_affinity))
	else:
		not_in_file +=1

print("Num alleles found: ", in_file)
print("Num alleles not found: ", not_in_file)

# Generate the new dataset with most of these amino acid sequences
with open("mhcflurry_dataset.json", 'w') as f:
	json.dump(comprehensive_dataset, f)
