# app.py
import streamlit as st
import time
import random

# Dummy Movie Title
movie_titles = [
    "The Shawshank Redemption", "The Godfather", "The Dark Knight", "Pulp Fiction",
    "Forrest Gump", "Inception", "Fight Club", "The Matrix", "Interstellar",
    "Gladiator", "Parasite", "Titanic", "Joker", "Avengers: Endgame",
    "The Lion King", "Whiplash", "Coco", "Toy Story", "The Prestige", "Up"
]

# Fake movie dataset
movies_df = [{"title": title} for title in movie_titles]

# Dummy models
svd_model = "dummy_svd_model"
cosine_sim_model = "dummy_cosine_model"

# Dummy Recommendation Function
def get_hybrid_recommendations(user_id, selected_movies, svd_model, cosine_sim_model, movies_df):
    """Simulates hybrid recommendations by randomly picking 10 movies not in user's favorites."""
    all_titles = [m["title"] for m in movies_df]
    possible_recs = [m for m in all_titles if m not in selected_movies]
    random.shuffle(possible_recs)
    recommended_movies = [
        {"title": t, "explanation": f"Because you liked movies similar to '{random.choice(selected_movies)}'."}
        for t in possible_recs[:10]
    ]
    return recommended_movies


# --- Header Section ---
st.title("🎬 CineMatch Hybrid Recommender")
st.markdown("""
Welcome to **CineMatch**! This advanced recommendation system combines two powerful techniques 
to give you truly personalized movie suggestions. 
Tell us who you are and what you love, and we'll find your next favorite movie.
""")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Tell Us About Yourself")
    st.markdown("To get the best recommendations, we need two things:")

    # 1. User Identifier Input
    user_id = st.number_input(
        "1. Enter Your User ID", 
        min_value=1, 
        max_value=10000, 
        value=1, 
        help="This helps us find what people with similar tastes have enjoyed (Collaborative Filtering)."
    )

    # 2. Favorite Movies Input
    selected_movies = st.multiselect(
        "2. Select at least 5 of your favorite movies",
        movie_titles,
        max_selections=10,
        help="This tells us what kind of movies you like based on their content (Content-Based Filtering)."
    )

    # Recommendation Button
    st.markdown("---")
    recommend_button = st.button("✨ Get My Recommendations", type="primary", use_container_width=True)


# --- Recommendation Logic and Display ---
if recommend_button:
    if not user_id:
        st.error("Please enter a User ID.")
    elif len(selected_movies) < 5:
        st.warning("Please select at least 5 favorite movies to get the best results.")
    elif svd_model is None or cosine_sim_model is None or not movies_df:
        st.error("Models or data are not loaded. Cannot generate recommendations.")
    else:
        with st.spinner('🧠 Analyzing your taste and finding the perfect movies...'):
            time.sleep(2)
            recommended_movies = get_hybrid_recommendations(
                user_id, selected_movies, svd_model, cosine_sim_model, movies_df
            )

        if recommended_movies:
            st.success(f"Here are your top 10 personalized recommendations, User {user_id}!")

            # Display recommendations in 2 columns for readability
            num_cols = 2
            cols = st.columns(num_cols)
            for i, movie in enumerate(recommended_movies):
                with cols[i % num_cols]:
                    st.subheader(f"🎥 {movie['title']}")
                    st.caption(movie["explanation"])
        else:
            st.info("Could not generate recommendations. Please try different movies or check the logs.")


# --- Footer/About Section ---
st.markdown("---")
st.markdown("""
**About this App:**
This is a demonstration of a hybrid recommendation system built with Python and Streamlit.  
It combines **Content-Based Filtering** (analyzing movie features) and **Collaborative Filtering** (analyzing user ratings) 
to provide nuanced and personalized suggestions.
""")
