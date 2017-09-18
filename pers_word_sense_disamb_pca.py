from nltk.corpus import wordnet as wn
from itertools import product
import numpy as np
import scipy as sp
from sklearn.decomposition import PCA

# min_sim = 0.1 # define the lowest acceptable path similarity
component_cutoff = 0.05 # define the lowest acceptable component loading
                       # see text

seeds = [('anger', 'n'), ('irritation', 'n'), ('temper', 'n')]
# list of tuples
# seed word, pos

seed_sets = [wn.synsets(x[0], x[1]) for x in seeds]
# returns a list of lists (L > l)
# each l is a Synset, see WordNet documentation (l > syn)

seed_sets_l = list(set([syn for l in seed_sets for syn in l]))
# flatten L
# returns a list of Synsets (L > syn)

seed_sets_indices = np.arange(len(seed_sets_l))
# an array of the indices of the syn's in L

sense_indices_cross_prod = product(seed_sets_indices, seed_sets_indices)
# the cross-product of syn's in L indices with themselves


ssl = seed_sets_l
# shorthand for L

seed_triplets = [(i, j, ssl[i].path_similarity(ssl[j])) for i,j in sense_indices_cross_prod] 
# not filtered yet
# return a triplet (row index, column index, similarity value)
# for each pair in the cross product

# seed_triplets_filtered = filter(lambda x: x[2] > min_sim, seed_triplets)

# the following lines prepare the data format for
# creating the adjacency matrix
rows = np.array(seed_triplets)[:,0]
cols = np.array(seed_triplets)[:,1]
vals = np.array(seed_triplets)[:,2]

wn_similarity_mx = sp.sparse.coo_matrix((vals, (rows, cols))).todense()
# create the adjacency matrix

# principal components
pca = PCA(n_components=1)
pca.fit(wn_similarity_mx)

sense_loadings = pca.components_[0]
sense_indices = np.array(np.where(sense_loadings > component_cutoff)).tolist()[0]

# disambiguated senses
seeds_final = list(seed_sets_l[i] for i in sense_indices)
seeds_final

synonyms = [x.name() for syn in seeds_final for x in syn.lemmas()]
synonyms


