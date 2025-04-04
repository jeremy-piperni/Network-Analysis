import numpy as np
import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
from itertools import combinations

# Get the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.join(script_dir, os.pardir, "Data", "author_genre_network.csv")
new_path = os.path.abspath(new_path)
df = pd.read_csv(new_path)

B = nx.Graph()

for _, row in df.iterrows():
    if isinstance(row["Mapped Genres"], str):
        author = row["Author"]
        genres = row["Mapped Genres"].split(', ')

        B.add_node(author, bipartite=0)
        for genre in genres:
            B.add_node(genre, bipartite=1)
            B.add_edge(author, genre)

author_nodes = {node for node, data in B.nodes(data=True) if data["bipartite"] == 0}
genre_nodes = set(B) - author_nodes

print("# of Author Nodes: " + str(len(author_nodes)))
print("# of Genre Nodes: " + str(len(genre_nodes)))
print("# of Edges: " + str(B.number_of_edges()))

largest_cc = max(nx.connected_components(B), key=len)
B_sub = B.subgraph(largest_cc)

node_colors = ["lawngreen" if node in author_nodes else "firebrick" for node in B_sub.nodes]

pos = nx.spring_layout(B_sub, seed=9)
nx.draw(B_sub, pos, node_size=10, width=0.5, node_color=node_colors, alpha=0.8)
plt.show()
