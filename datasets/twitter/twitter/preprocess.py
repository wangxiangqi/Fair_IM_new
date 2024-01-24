import networkx as nx
import pickle
import numpy as np
# Create an empty ded graph
from sklearn import preprocessing
graph = nx.DiGraph()

# Open the text file and read the edges
with open('428333.edges', 'r') as f:
    edges = f.readlines()

# Add the edges to the graph
for edge in edges:
    source, target = edge.strip().split()
    graph.add_edge(source, target, weight=1)

# Save the graph in .G file format
N = graph.number_of_nodes() + 0
mapping = dict(zip(graph.nodes(), range(0, N)))
#print(list(mapping.keys()))
graph= nx.convert_node_labels_to_integers(graph, label_attribute='pid')
with open('twitter_relationship.G', 'wb') as f:
    pickle.dump(graph, f)

import csv