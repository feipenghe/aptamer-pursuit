#!/usr/bin/env python
# coding: utf-8

# # Progress Until 05/14/2020

# ## ML Models

# ### We can't change lambda because that changes the objective function and we can't compare losses/recalls

# * 05022020.pth model (has lambda of 4, 3, 2)

# <img src="0502_loss.png">

# <img src="0502_recall.png">

# <img src="0502_model.png">

# * 05112020.pth (lambda=3, varying gamma)

# <img src="../src/models/plots/mle/MinimizedVCNet/05112020/histogram.png">

# <img src="../src/models/plots/mle/MinimizedVCNet/05112020/test_loss.png">

# <img src="../src/models/plots/mle/MinimizedVCNet/05112020/train_loss.png">

# ### Next Steps: 
# * We have batch sizes! Now our models train a little more quickly
# * Kevin wants to see a model that overfits --> maybe move into ResNet territory

# ## Similarity between predictions and the train set
# * 2 points for matching, -1 points for mismatch, -2 for opening gap, -0.5 for continuing a gap

# * Generated and existing aptamers to train set

# <img src="similarity-proper-aptamers.png">

# * Generated and existing peptides to train set

# <img src="similarity-proper-peptides.png">

# ## mFold

# ### In general, this doesn't work for the following reasons:
# * Unafold, which is theoretically better (but still has the same usability issues) requires a fee to use. 
# * mFold doens't work on mac, it has to be put on linux. 
# * Even when it's on linux, the scripts are buggy. It's taken me > 3 hours to deal with bugs and I haven't been able to debug them. 
# * The scripts still require that you only read per one sequence even if you generate a fasta file for all sequences.
# * Maybe there's a better way to understand biological relevance

# ### Biological relevance question

# In[8]:


## Saliency maps (similar to how it's done in images)


# In[9]:


## But can we do it for sequences?


# In[ ]:




