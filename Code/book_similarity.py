import numpy as np
import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import community as community_louvain

# Get the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.join(script_dir, os.pardir, "Data", "book_description_network.csv")
new_path = os.path.abspath(new_path)
df = pd.read_csv(new_path)
df = df.dropna(subset=["Title", "Description"])

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["Description"])

cosine_sim = cosine_similarity(tfidf_matrix)

G = nx.Graph()

for book in df["Title"]:
    G.add_node(book)

threshold = 0.5
for i in range (len(df)):
    for j in range(i+1, len(df)):
        sim_score = cosine_sim[i, j]
        if sim_score >= threshold:
            G.add_edge(df["Title"][i], df["Title"][j], weight=sim_score)

G.remove_edges_from(nx.selfloop_edges(G))

print("# of Nodes: " + str(G.number_of_nodes()))
print("# of Edges: " + str(G.number_of_edges()))

# Code to draw the network
'''
pos = nx.spring_layout(G, seed=9)
nx.draw(G, pos, node_size=20, width=0.5, node_color='deepskyblue', alpha=0.8)
plt.show()
'''

# Compute louvain community detection, remove all single nodes as they do not belong to a community
G.remove_nodes_from(list(nx.isolates(G)))
partition = community_louvain.best_partition(G)

# Get the book-genre information
script_dir = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.join(script_dir, os.pardir, "Data", "genre_book_network.csv")
new_path = os.path.abspath(new_path)
genre_df = pd.read_csv(new_path)

# Add communities to genre dataframe and get the genres for each book
genre_df["Community"] = genre_df["Title"].map(partition)
genre_df = genre_df.dropna(subset=["Community", "Mapped Genres"])
genre_df["Genres"] = genre_df["Mapped Genres"].str.split(', ')

# Count genres per community and track size
community_genre_dist = defaultdict(Counter)
community_sizes = defaultdict(int)
for _, row in genre_df.iterrows():
    community_id = int(row["Community"])
    genres = row["Genres"]
    community_genre_dist[community_id].update(genres)
    community_sizes[community_id] += 1

# Compute % distributions and print sorted results
print(f"Number of Communities: {len(set(partition.values()))}")
for community_id, genre_counter in community_genre_dist.items():
    total = sum(genre_counter.values())
    genre_percentages = {genre: (count / total) * 100 for genre, count in genre_counter.items()}
    sorted_genres = sorted(genre_percentages.items(), key=lambda x: x[1], reverse=True)
    if community_sizes[community_id] >= 50:
        print(f"\n Community {community_id} (Size: {community_sizes[community_id]}) - Genre Distribution (%):")
        for genre, pct in sorted_genres:
            if pct >= 5:
                print(f" {genre}: {pct:.2f}%")

# Code to draw the louvain community network
'''
pos = nx.spring_layout(G)
communities = set(partition.values())
cmap = cm.get_cmap('viridis', len(communities))
for community_id in communities:
    nodes = [node for node in partition if partition[node] == community_id]
    nx.draw_networkx_nodes(G, pos, node_size=10, alpha=0.8, nodelist=nodes, node_color=[cmap(community_id)])
nx.draw_networkx_edges(G, pos, width=0.5)
plt.show()
'''