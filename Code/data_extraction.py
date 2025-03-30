import numpy as np
import pandas as pd
import os
import genre_mapping

# Get the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.join(script_dir, os.pardir, "Data", "merged_dataset.csv")
new_path = os.path.abspath(new_path)
df = pd.read_csv(new_path)

# Remove all books that are uncensored
df = df.loc[df["Banned"] != 0]

# Gather the unique genres while cleaning and sorting
unique_genres = set(df["Genre"].astype(str).fillna("").str.split(", ").explode())
cleaned_genres = {word.lower().removeprefix("null") for word in unique_genres}
sorted_genres = sorted(cleaned_genres)

print("# of Genres: " + str(len(sorted_genres)))

# Apply the genre mapping
new_genres = set()
for word in sorted_genres:
    new_genres.add(genre_mapping.map_genre(word))

print("# of Genres after Mapping: " + str(len(new_genres)))

# Function to return the mapped genres in the dataframe
def map_words(text):
    if not isinstance(text, str):
        return text
    words = text.split(", ")
    words = [word.lower().removeprefix("null") for word in words]
    mapped_words = [genre_mapping.map_genre(word) for word in words]
    return ', '.join(mapped_words)

# Get the mapped genres, remove old genre column
df["Mapped Genres"] = df["Genre"].apply(lambda x: map_words(x))
df = df.drop(columns=["Genre"])

# Export dataset1, Title and Genre
df1 = df.drop(columns=["Author","Description","Banned"])
script_dir1 = os.path.dirname(os.path.abspath(__file__))
new_path1 = os.path.join(script_dir1, os.pardir, "Data", "final_dataset1.csv")
new_path1 = os.path.abspath(new_path1)
df1.to_csv(new_path1, index=False)

