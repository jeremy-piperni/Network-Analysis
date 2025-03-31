import numpy as np
import pandas as pd
import os
import genre_mapping
import re

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

# Function to format the author names properly
def format_author(author):
    author = author.lower().strip()
    author = author.encode('ascii', 'ignore').decode()
    if ' ' not in author:
        return author
    else:
        author = re.split(r',?\s+', author) 
        if len(author) == 2:
            if len(author[0]) < len(author[1]):
                return f"{author[0]} {author[1]}"
            else:
                return f"{author[1]} {author[0]}"
        if len(author) == 3:
            if author[2].endswith('.'):
                if author[1].endswith('.'):
                    return f"{author[1]} {author[2]} {author[0]}"
                else:
                    return f"{author[2]} {author[0]} {author[1]}"
    
    return ' '.join(author)

# Function to format descriptions
def format_description(description):
    description = description.lower().strip()
    description = description.encode('ascii', 'ignore').decode()
    return description

# Get the mapped genres, remove old genre column
df["Mapped Genres"] = df["Genre"].apply(lambda x: map_words(x))
df = df.drop(columns=["Genre"])

# Clean the author column
df["Author"] = df["Author"].astype(str).apply(format_author)

# Clean the description column
df["Description"] = df["Description"].astype(str).apply(format_description)

# Export Genre-Book network data
df1 = df.drop(columns=["Author","Description","Banned"])
script_dir1 = os.path.dirname(os.path.abspath(__file__))
new_path1 = os.path.join(script_dir1, os.pardir, "Data", "genre_book_network.csv")
new_path1 = os.path.abspath(new_path1)
df1.to_csv(new_path1, index=False)

# Export Author-Book network data
df2 = df.drop(columns=["Mapped Genres","Description","Banned"])
script_dir2 = os.path.dirname(os.path.abspath(__file__))
new_path2 = os.path.join(script_dir2, os.pardir, "Data", "author_book_network.csv")
new_path2 = os.path.abspath(new_path2)
df2.to_csv(new_path2, index=False)

# Export Author-Genre network data
df3 = df.drop(columns=["Title","Description","Banned"])
script_dir3 = os.path.dirname(os.path.abspath(__file__))
new_path3 = os.path.join(script_dir3, os.pardir, "Data", "author_genre_network.csv")
new_path3 = os.path.abspath(new_path3)
df3.to_csv(new_path3, index=False)

# Export Book-Description network data
df4 = df.drop(columns=["Mapped Genres","Author","Banned"])
script_dir4 = os.path.dirname(os.path.abspath(__file__))
new_path4 = os.path.join(script_dir4, os.pardir, "Data", "book_description_network.csv")
new_path4 = os.path.abspath(new_path4)
df4.to_csv(new_path4, index=False)
