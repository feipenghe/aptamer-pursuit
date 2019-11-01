# Model: n-gram

### Experimental Process:
I created a dataset that maps every allele name to a list of peptides it will bind to (the binding affinity doesn't make a difference here). Based on
that dataset, I predicted one protein that will bind for each allele. The code for this experiment is in src/models/ngram.py.

### Results:
I can generate a set of predicted peptides, but I don't have a way to validate them in any way now. I can potentially use MHCSeqNet
to validate my results and give me a better way of predicting future peptides. 

### Next Steps:
1. Use MHCSeqNet to output a binding prediction. 
2. Try more complex networks that incorporate the binding affinity data. 
3. Analyze the predicted proteins to understand what motifs I'm picking up on in an n-gram model. Maybe that will give me an indication of how to proceed. 

