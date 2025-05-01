from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict
import pandas as pd

class CollaborativeRecommender:
    def __init__(self, ratings):
        # Prepare Surprise dataset
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
        self.trainset, self.testset = train_test_split(data, test_size=0.2, random_state=42)
        self.model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
        self.model.fit(self.trainset)
    
    def evaluate(self):
        """Evaluate model performance"""
        predictions = self.model.test(self.testset)
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        return rmse, mae
    
    def get_user_recommendations(self, user_id, movies, top_n=10):
        """Get top recommendations for a user"""
        # Get list of all movie IDs
        all_movie_ids = movies['movie_id'].unique()
        
        # Get list of movie IDs the user has already rated
        try:
            user_inner_id = self.trainset.to_inner_uid(user_id)
            rated_movies = self.trainset.ur[user_inner_id]
            rated_movie_ids = [self.trainset.to_raw_iid(i[0]) for i in rated_movies]
        except ValueError:  # User not in trainset
            rated_movie_ids = []
        
        # Get list of movie IDs not rated by the user
        unrated_movie_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids]
        
        # Predict ratings for unrated movies
        testset = [[user_id, movie_id, 4.] for movie_id in unrated_movie_ids]
        predictions = self.model.test(testset)
        
        # Get top N recommendations
        top_n = defaultdict(list)
        for uid, mid, true_r, est, _ in predictions:
            top_n[uid].append((mid, est))
        
        # Sort the predictions for each user and retrieve the top N
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:top_n]
        
        # Convert to DataFrame
        recommendations = []
        for movie_id, rating in top_n[user_id]:
            movie_info = movies[movies['movie_id'] == movie_id][['movie_id', 'title', 'genres']].iloc[0]
            recommendations.append({
                'movie_id': movie_id,
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'predicted_rating': rating
            })
        
        return pd.DataFrame(recommendations)
    
    def predict_rating(self, user_id, movie_id):
        """Predict rating for a user-movie pair"""
        return self.model.predict(user_id, movie_id).est