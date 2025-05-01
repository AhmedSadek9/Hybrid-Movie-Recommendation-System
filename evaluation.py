import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

def evaluate_models(content_rec, collab_rec, hybrid_rec, full_data, movies, ratings):
    """Evaluate all recommendation approaches"""
    # Collaborative filtering metrics
    rmse, mae = collab_rec.evaluate()
    print(f"Collaborative Filtering - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # Content-based filtering analysis
    sample_movie = movies.sample(1)['title'].values[0]
    print(f"\nContent-Based Recommendations for '{sample_movie}':")
    print(content_rec.get_recommendations(sample_movie, top_n=3))
    
    # Hybrid approach analysis
    sample_user = ratings['user_id'].sample(1).values[0]
    user_ratings = dict(zip(
        full_data[full_data['user_id'] == sample_user]['title'],
        full_data[full_data['user_id'] == sample_user]['rating']
    ))
    print(f"\nHybrid Recommendations for User {sample_user}:")
    print(hybrid_rec.recommend(
        user_id=sample_user,
        user_ratings=user_ratings,
        top_n=3,
        content_weight=0.4,
        collab_weight=0.6
    ))
    
    # Diversity analysis
    all_recs = []
    for user in ratings['user_id'].sample(10).values:
        user_ratings = dict(zip(
            full_data[full_data['user_id'] == user]['title'],
            full_data[full_data['user_id'] == user]['rating']
        ))
        recs = hybrid_rec.recommend(user_id=user, user_ratings=user_ratings, top_n=5)
        all_recs.extend(recs['title'].tolist())
    
    rec_counts = pd.Series(all_recs).value_counts().head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=rec_counts.values, y=rec_counts.index)
    plt.title("Top 10 Most Frequently Recommended Movies")
    plt.xlabel("Number of Recommendations")
    plt.show()