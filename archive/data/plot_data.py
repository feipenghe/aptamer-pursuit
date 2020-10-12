import matplotlib.pyplot as plt
import numpy as np
# Load the text data
bdata_file = "bdata.20130222.mhci.txt"

binding_affinities = list()
file = open(bdata_file)
data = file.readlines()[1:]
for line in data:
	parsed = line.split()
	allele = parsed[1]
	peptide_seq = parsed[3]
	binding_affinity = float(parsed[5])

	binding_affinities.append(binding_affinity)

print(binding_affinities[0:20])

# index = [i for i in range(len(binding_affinities))]
num_bins = 100
plt.hist(np.log(binding_affinities), num_bins) #, normed=1)
plt.show()

