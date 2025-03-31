import numpy as np
import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
from itertools import combinations

# Get the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.join(script_dir, os.pardir, "Data", "genre_book_network.csv")
new_path = os.path.abspath(new_path)
df = pd.read_csv(new_path)

G = nx.Graph()

for genres in df["Mapped Genres"]:
    if isinstance(genres, str):
        genre_list = genres.split(', ')
        for genre1, genre2 in combinations(genre_list, 2):
            if G.has_edge(genre1, genre2):
                G[genre1][genre2]['weight'] += 1
            else:
                G.add_edge(genre1, genre2, weight = 1)

print("# of Nodes: " + str(G.number_of_nodes()))
print("# of Edges: " + str(G.number_of_edges()))

pos = nx.spring_layout(G, seed=9)
nx.draw(G, pos, node_size=20, width=0.5, node_color='firebrick', alpha=0.8)
plt.show()
