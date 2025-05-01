import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, movies):
        self.movies = movies.reset_index(drop=True)
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(movies['genres'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    
    def get_recommendations(self, title, top_n=10):
        """Get recommendations based on movie title"""
        try:
            idx = self.indices[title]
        except KeyError:
            return pd.DataFrame()
        
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]  # Skip the first item (itself)
        
        movie_indices = [i[0] for i in sim_scores]
        recommendations = self.movies.iloc[movie_indices][['movie_id', 'title', 'genres']]
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        
        return recommendations
    
    def get_user_profile_recommendations(self, user_ratings, top_n=10):
        """Get recommendations based on user's rated movies"""
        if not user_ratings:
            return pd.DataFrame()
            
        # Calculate weighted average of similarity scores
        total_scores = np.zeros(len(self.movies))
        total_weight = 0
        
        for movie_title, rating in user_ratings.items():
            try:
                idx = self.indices[movie_title]
                weight = (rating - 2.5) * 2  # Scale to give more weight to higher ratings
                total_scores += self.cosine_sim[idx] * weight
                total_weight += abs(weight)
            except KeyError:
                continue
                
        if total_weight == 0:
            return pd.DataFrame()
            
        avg_scores = total_scores / total_weight
        
        # Get top recommendations
        sim_scores = list(enumerate(avg_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[:top_n]
        
        movie_indices = [i[0] for i in sim_scores]
        recommendations = self.movies.iloc[movie_indices][['movie_id', 'title', 'genres']]
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        
        return recommendations