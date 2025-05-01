import streamlit as st
from data_processing import download_movielens_data, load_data, preprocess_data
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from hybrid import HybridRecommender
import pandas as pd

# Download and load data
download_movielens_data()
ratings, movies, users = load_data()
full_data, movies = preprocess_data(ratings, movies, users)

# Initialize recommenders
content_rec = ContentBasedRecommender(movies)
collab_rec = CollaborativeRecommender(ratings)
hybrid_rec = HybridRecommender(content_rec, collab_rec, movies)

# Streamlit app
st.title("Hybrid Movie Recommendation System")

# Sidebar for navigation
option = st.sidebar.radio("Choose recommendation type:", 
                         ["Movie-Based Recommendations", "User-Based Recommendations"])

if option == "Movie-Based Recommendations":
    st.header("Get recommendations based on a movie you like")
    
    movie_title = st.selectbox("Select a movie:", movies['title'].values)
    
    if st.button("Get Recommendations"):
        st.subheader(f"Recommendations similar to '{movie_title}':")
        
        # Get content-based recommendations
        content_recs = content_rec.get_recommendations(movie_title, top_n=10)
        
        if not content_recs.empty:
            st.write("Content-Based Recommendations:")
            st.dataframe(content_recs[['title', 'genres', 'similarity_score']].rename(
                columns={'similarity_score': 'Similarity Score'}
            ))
        else:
            st.warning("No recommendations found for this movie.")

else:
    st.header("Get personalized recommendations based on your ratings")
    
    user_id = st.number_input("Enter your user ID (1-943):", min_value=1, max_value=943, value=1)
    
    # Get user's rated movies
    user_ratings = full_data[full_data['user_id'] == user_id]
    user_ratings_dict = dict(zip(user_ratings['title'], user_ratings['rating']))
    
    if st.button("Get Personalized Recommendations"):
        st.subheader(f"Personalized Recommendations for User {user_id}:")
        
        # Get hybrid recommendations
        hybrid_recs = hybrid_rec.recommend(
            user_id=user_id,
            user_ratings=user_ratings_dict,
            top_n=10,
            content_weight=0.4,
            collab_weight=0.6
        )
        
        if not hybrid_recs.empty:
            st.write("Hybrid Recommendations:")
            st.dataframe(hybrid_recs[['title', 'genres', 'hybrid_score']].rename(
                columns={'hybrid_score': 'Recommendation Score'}
            ))
            
            # Show user's rated movies
            st.subheader("Your Rated Movies:")
            st.dataframe(user_ratings[['title', 'rating', 'genres']].rename(
                columns={'rating': 'Your Rating'}
            ))
        else:
            st.warning("No recommendations found for this user.")

# Add some information about the system
st.sidebar.markdown("""
### About This System
This hybrid recommendation system combines:
- **Content-Based Filtering**: Recommends similar movies based on genre similarity
- **Collaborative Filtering**: Recommends movies based on ratings from similar users
""")





#streamlit run app.py