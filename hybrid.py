import pandas as pd
import numpy as np

class HybridRecommender:
    def __init__(self, content_recommender, collab_recommender, movies):
        self.content_rec = content_recommender
        self.collab_rec = collab_recommender
        self.movies = movies
    
    def recommend(self, user_id=None, user_ratings=None, title=None, top_n=10, content_weight=0.5, collab_weight=0.5):
        """Generate hybrid recommendations"""
        if user_id is not None and user_ratings is not None:
            # Get content-based recommendations based on user's rated movies
            content_recs = self.content_rec.get_user_profile_recommendations(user_ratings, top_n*2)
            
            # Get collaborative recommendations
            collab_recs = self.collab_rec.get_user_recommendations(user_id, self.movies, top_n*2)
        elif title is not None:
            # Get content-based recommendations for a specific movie
            content_recs = self.content_rec.get_recommendations(title, top_n*2)
            collab_recs = pd.DataFrame()
        else:
            return pd.DataFrame()
        
        # Normalize scores
        if not content_recs.empty:
            content_recs['normalized_score'] = (content_recs['similarity_score'] - content_recs['similarity_score'].min()) / \
                                            (content_recs['similarity_score'].max() - content_recs['similarity_score'].min())
        
        if not collab_recs.empty:
            collab_recs['normalized_score'] = (collab_recs['predicted_rating'] - collab_recs['predicted_rating'].min()) / \
                                            (collab_recs['predicted_rating'].max() - collab_recs['predicted_rating'].min())
        
        # Merge recommendations
        if not content_recs.empty and not collab_recs.empty:
            merged_recs = pd.merge(
                content_recs, 
                collab_recs, 
                on=['movie_id', 'title', 'genres'], 
                how='outer',
                suffixes=('_content', '_collab')
            )
            
            # Fill NaN values with 0
            merged_recs['normalized_score_content'] = merged_recs['normalized_score_content'].fillna(0)
            merged_recs['normalized_score_collab'] = merged_recs['normalized_score_collab'].fillna(0)
            
            # Calculate hybrid score
            merged_recs['hybrid_score'] = (content_weight * merged_recs['normalized_score_content'] + 
                                         collab_weight * merged_recs['normalized_score_collab'])
            
            # Sort by hybrid score
            final_recs = merged_recs.sort_values('hybrid_score', ascending=False).head(top_n)
            
        elif not content_recs.empty:
            final_recs = content_recs.sort_values('similarity_score', ascending=False).head(top_n)
            final_recs['hybrid_score'] = final_recs['similarity_score']
        elif not collab_recs.empty:
            final_recs = collab_recs.sort_values('predicted_rating', ascending=False).head(top_n)
            final_recs['hybrid_score'] = final_recs['predicted_rating']
        else:
            return pd.DataFrame()
        
        return final_recs[['movie_id', 'title', 'genres', 'hybrid_score']]