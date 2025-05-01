import pandas as pd
import numpy as np
import os
import zipfile
from urllib.request import urlretrieve

def download_movielens_data():
    """Download and extract MovieLens 100K dataset"""
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/ml-100k"):
        print("Downloading MovieLens 100K dataset...")
        urlretrieve(url, "data/ml-100k.zip")
        with zipfile.ZipFile("data/ml-100k.zip", 'r') as zip_ref:
            zip_ref.extractall("data")
        os.remove("data/ml-100k.zip")
        print("Download complete!")

def load_data():
    """Load MovieLens 100K dataset"""
    # Load ratings data
    ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', 
                         names=['user_id', 'movie_id', 'rating', 'timestamp'], 
                         encoding='latin-1')
    
    # Load movie data
    movies = pd.read_csv('data/ml-100k/u.item', sep='|', 
                        names=['movie_id', 'title', 'release_date', 'video_release_date',
                               'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                               'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                               'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                               'Thriller', 'War', 'Western'], 
                        encoding='latin-1')
    
    # Load user data
    users = pd.read_csv('data/ml-100k/u.user', sep='|', 
                       names=['user_id', 'age', 'gender', 'occupation', 'zip_code'], 
                       encoding='latin-1')
    
    return ratings, movies, users

def preprocess_data(ratings, movies, users):
    """Preprocess the datasets"""
    # Create genre string for content-based filtering
    genre_cols = movies.columns[6:]
    movies['genres'] = movies[genre_cols].apply(lambda x: ' '.join(x.index[x==1]), axis=1)
    
    # Clean movie titles (remove year)
    movies['title'] = movies['title'].str.replace(r'\(\d{4}\)', '').str.strip()
    
    # Merge datasets
    movie_ratings = pd.merge(ratings, movies, on='movie_id')
    full_data = pd.merge(movie_ratings, users, on='user_id')
    
    return full_data, movies