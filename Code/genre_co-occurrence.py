import numpy as np
import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import combinations
import community as community_louvain

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

# Code to draw the network
'''
pos = nx.spring_layout(G, seed=9)
nx.draw(G, pos, node_size=20, width=0.5, node_color='firebrick', alpha=0.8)
plt.show()
'''

# Get weighted betweenenss centrality scores
genre_bet_centrality = nx.betweenness_centrality(G, weight='weight')
top_genre_bet = sorted(genre_bet_centrality.items(), key=lambda x: x[1], reverse=True)[:15]
for node, centrality in top_genre_bet:
    print(f"Genre Node: {node}, Betweeness Centrality: {round(centrality,4)}")

# Get weighted closeness centrality scores
genre_close_centrality = nx.closeness_centrality(G, distance='weight')
top_genre_close = sorted(genre_close_centrality.items(), key=lambda x: x[1], reverse=True)[:15]
for node, centrality in top_genre_close:
    print(f"Genre Node: {node}, Closeness Centrality: {round(centrality,4)}")

# Compute louvain community detection
partition = community_louvain.best_partition(G)

# Code to analyze which genre is in which community
'''
ammount = 0
for node, community_id in partition.items():
    if community_id == 1:
        ammount += 1
        print(f"{node}: {community_id}")
print(ammount)
'''

# Code to draw the louvain community network
pos = nx.spring_layout(G)
communities = set(partition.values())
cmap = cm.get_cmap('viridis', len(communities))
for community_id in communities:
    nodes = [node for node in partition if partition[node] == community_id]
    nx.draw_networkx_nodes(G, pos, node_size=10, alpha=0.8, nodelist=nodes, node_color=[cmap(community_id)])
nx.draw_networkx_edges(G, pos, width=0.5)
plt.show()
