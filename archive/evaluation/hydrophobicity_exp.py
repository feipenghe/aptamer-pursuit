import os, sys, json

dataset_file = "data/mhcflurry_dataset.json"

with open(dataset_file, 'r') as f:
	data = json.load(f)

hydrophobic = ['W', 'I', 'L', 'V', 'F', 'Y']

affinities = []
negative_affinities = []
for allele in data:
	index = 0
	for a in allele:
		if a in hydrophobic:
			index += 1

	index /= len(allele)

	proteins = data[allele]
	for i, p in enumerate(proteins):
		seq, aff = p
		ind = 0
		for s in seq:
			if s in hydrophobic:
				ind += 1
		ind /= len(seq)

		if index > 0.5 and ind > 0.5:
			print("Allele: " + str(allele) + " Binding Affinity: " + aff)
			affinities.append(float(aff))
		else:
			negative_affinities.append(float(aff))


