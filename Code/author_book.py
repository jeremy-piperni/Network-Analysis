import numpy as np
import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
from itertools import combinations
from networkx.algorithms import bipartite

# Get the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.join(script_dir, os.pardir, "Data", "author_book_network.csv")
new_path = os.path.abspath(new_path)
df = pd.read_csv(new_path)

B = nx.Graph()

for _, row in df.iterrows():
    book = row["Title"]
    author = row["Author"]

    B.add_node(book, bipartite=0)
    B.add_node(author, bipartite=1)
    B.add_edge(book, author)

book_nodes = {node for node, data in B.nodes(data=True) if data["bipartite"] == 0}
author_nodes = set(B) - book_nodes

print("# of Book Nodes: " + str(len(book_nodes)))
print("# of Author Nodes: " + str(len(author_nodes)))
print("# of Edges: " + str(B.number_of_edges()))

# Code to draw the network
'''
node_colors = ["deepskyblue" if node in book_nodes else "lawngreen" for node in B.nodes]

pos = nx.spring_layout(B, seed=9)
nx.draw(B, pos, node_size=10, width=0.5, node_color=node_colors, alpha=0.8)
plt.show()
'''

# Get highest degree centrality scores
author_deg_centrality = {n: c for n, c in bipartite.degree_centrality(B, author_nodes).items() if n in author_nodes}
top_author_deg = sorted(author_deg_centrality.items(), key=lambda x: x[1], reverse=True)[:15]
for node, centrality in top_author_deg:
    print(f"Author Node: {node}, Deg Centrality: {round(centrality,4)}")

# Perform eigenvector centrality on the author nodes
author_projection = bipartite.projected_graph(B, author_nodes)
author_eigen_centrality = nx.eigenvector_centrality(author_projection)
top_author_eigen = sorted(author_eigen_centrality.items(), key=lambda x: x[1], reverse=True)[:15]
for node, centrality in top_author_eigen:
    print(f"Author Node: {node}, Eigen Centrality: {round(centrality,4)}")