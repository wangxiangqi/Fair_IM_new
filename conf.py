import os
from Oracle.generalGreedy import generalGreedy
from Oracle.degreeDiscount import degreeDiscountIC, degreeDiscountIC2, degreeDiscountIAC, degreeDiscountIAC2, degreeDiscountStar, degreeDiscountIAC3
from Oracle.Fair_Oracle import Fair_IM_oracle, optimal_gred


save_address = "./SimulationResults"

#graph_address = './datasets/Flickr/Small_Final_SubG.G'
#prob_address = './datasets/Flickr/Probability.dic'
#param_address = './datasets/Flickr/Small_nodeFeatures.dic'
#edge_feature_address = './datasets/Flickr/Small_edgeFeatures.dic'
#dataset = 'Flickr-Random' #Choose from 'default', 'NetHEPT', 'Flickr'

# graph_address = './datasets/Flickr/G_Union.G'
# prob_address = './datasets/Flickr/ProbUnion.dic'
# param_address = './datasets/Flickr/NodeFeaturesUnion.dic'
# edge_feature_address = './datasets/Flickr/EdgeFeaturesUnion.dic'
# dataset = 'Flickr-Cluster' #Choose from 'default', 'NetHEPT', 'Flickr'

# graph_address = './datasets/NetHEPT/Small_Final_SubG.G'
# prob_address = './datasets/NetHEPT/Probability.dic'
# param_address = './datasets/NetHEPT/Small_nodeFeatures.dic'
# edge_feature_address = './datasets/NetHEPT/Small_edgeFeatures.dic'
# dataset = 'NetHEPT' #Choose from 'default', 'NetHEPT', 'Flickr'
"""
graph_address = './datasets/NBA/nba_relationship.G'
prob_address = './datasets/NBA/prob.dic'
param_address = './datasets/NBA/parameter.dic'
edge_feature_address = './datasets/NBA/edge_feature.dic'
## For simulation on Fair IM, we need to choose a different dataset which includes class information
# Choose the third column of edge feature as class feature
class_feature_address = './datasets/NBA/country.dic'
dataset = 'NBA' #Choose from 'default', 'NetHEPT', 'Flickr'
"""

graph_address = './datasets/german/german_relationship.G'
prob_address = './datasets/german/german_prob.dic'
param_address = './datasets/german/parameter.dic'
edge_feature_address = './datasets/german/edge_feature.dic'
# For simulation on Fair IM, we need to choose a different dataset which includes class information
# Choose the third column of edge feature as class feature
class_feature_address = './datasets/german/Age.dic'
dataset = 'german' #Choose from 'default', 'NetHEPT', 'Flickr'

"""
graph_address = './datasets/pokec/pokec_relationship.G'
prob_address = './datasets/pokec/prob.dic'
param_address = './datasets/pokec/parameter.dic'
edge_feature_address = './datasets/pokec/edge_feature.dic'
### For simulation on Fair IM, we need to choose a different dataset which includes class information
# Choose the third column of edge feature as class feature
class_feature_address = './datasets/pokec/gender.dic'
#I made a mistake here, actually the distribution is age that I selected.
dataset = 'pokec' #Choose from 'default', 'NetHEPT', 'Flickr'
"""
alpha_1 = 0.1
alpha_2 = 0.1
lambda_ = 0.4
gamma = 0.1
dimension = 4
seed_size = 300
iterations = 100

oracle = degreeDiscountIAC3