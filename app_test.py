import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

# Konfigurasi Halaman
st.set_page_config(
    page_title="Rekomendasi Film Hibrida",
    page_icon="🎬",
    layout="wide"
)

# Path Model 
MODEL_DIR = 'models/'

# Fungsi Caching untuk Memuat Model 
@st.cache_resource
def load_models():
    try:
        movies_df = pickle.load(open(os.path.join(MODEL_DIR, 'movies_df.pkl'), 'rb'))
        ratings_df = pickle.load(open(os.path.join(MODEL_DIR, 'ratings_df.pkl'), 'rb'))
        svd_model = pickle.load(open(os.path.join(MODEL_DIR, 'svd_model.pkl'), 'rb'))
        tfidf_vectorizer = pickle.load(open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'rb'))
        tfidf_matrix = pickle.load(open(os.path.join(MODEL_DIR, 'tfidf_matrix.pkl'), 'rb'))
    except FileNotFoundError as e:
        st.error(f"Error: File model tidak ditemukan. Pastikan file .pkl ada di folder '{MODEL_DIR}'.")
        st.error(f"File yang hilang: {e.filename}")
        st.error("Pastikan Anda telah menjalankan notebook pipeline dan mengekspor model-model baru (termasuk tfidf_matrix.pkl).")
        return None, None, None, None, None
    print("Model berhasil dimuat.")
    return movies_df, ratings_df, svd_model, tfidf_vectorizer, tfidf_matrix


@st.cache_data # Cache perhitungan ini
def get_content_based_recommendations(liked_movie_titles, _movies_df, _tfidf_matrix, top_n=50):
    # Membuat series untuk mapping judul ke index
    indices = pd.Series(_movies_df.index, index=_movies_df['title'])
    # Mendapatkan index dari 5 film yang disukai
    liked_indices = [indices[title] for title in liked_movie_titles if title in indices]
    if not liked_indices: 
        return pd.DataFrame() # Kembalikan DataFrame kosong

    # PERHITUNGAN ON-THE-FLY
    # 1. Ambil vektor TF-IDF dari film yang disukai
    liked_vectors = _tfidf_matrix[liked_indices]
    # 2. Hitung cosine similarity ( (5, F) vs (N, F) ) -> (5, N)
    sim_scores_per_movie = cosine_similarity(liked_vectors, _tfidf_matrix)
    # 3. Rata-ratakan skornya -> (1, N)
    avg_sim_scores = sim_scores_per_movie.mean(axis=0)
    
    # Mengurutkan film berdasarkan skor similaritas rata-rata
    sim_scores = list(enumerate(avg_sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Mendapatkan skor dari top_n film
    sim_scores = sim_scores[1:top_n+len(liked_indices)] # +len untuk skip film yg sudah disukai
    
    # Mendapatkan index film
    movie_indices = [i[0] for i in sim_scores]
    
    # Mengambil judul film dan skor
    recs_df = _movies_df.iloc[movie_indices][['title', 'genres']]
    recs_df['score'] = [i[1] for i in sim_scores]
    
    # Menghapus film yang sudah disukai dari rekomendasi
    recs_df = recs_df[~recs_df['title'].isin(liked_movie_titles)]
    
    return recs_df.head(top_n)

@st.cache_data
def get_collaborative_filtering_recommendations(user_id, _svd_model, _movies_df, _ratings_df, top_n=50):
    """
    Memberikan rekomendasi collaborative filtering (SVD) untuk user_id.
    """
    # Mendapatkan daftar film yang belum ditonton oleh user
    movie_ids_watched = _ratings_df[_ratings_df['userId'] == user_id]['movieId'].unique()
    movie_ids_all = _movies_df['movieId'].unique()
    movie_ids_to_predict = np.setdiff1d(movie_ids_all, movie_ids_watched)
    
    # Membuat test set untuk prediksi
    testset = [[user_id, movie_id, 4.] for movie_id in movie_ids_to_predict]
    
    # Memprediksi rating
    predictions = _svd_model.test(testset)
    
    # Mengurutkan prediksi berdasarkan estimasi rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Mengambil top_n rekomendasi
    recs = []
    for pred in predictions[:top_n]:
        movie_id = pred.iid
        # Cek apakah movieId ada di movies_df
        movie_details = _movies_df[_movies_df['movieId'] == movie_id]
        if not movie_details.empty:
            movie_title = movie_details['title'].values[0]
            movie_genre = movie_details['genres'].values[0]
            recs.append({'title': movie_title, 'genres': movie_genre, 'score': pred.est})
        
    return pd.DataFrame(recs)

@st.cache_data
def get_hybrid_recommendations(user_id, liked_movie_titles, _movies_df, _tfidf_matrix, _svd_model, _ratings_df, top_n=10):
    """
    Menggabungkan rekomendasi Content-Based dan Collaborative Filtering.
    """
    # 1. Dapatkan Rekomendasi Content-Based (On-the-fly)
    cb_recs = get_content_based_recommendations(liked_movie_titles, _movies_df, _tfidf_matrix, top_n=50)
    
    # 2. Dapatkan Rekomendasi Collaborative Filtering
    cf_recs = get_collaborative_filtering_recommendations(user_id, _svd_model, _movies_df, _ratings_df, top_n=50)
    
    if cb_recs.empty and cf_recs.empty:
        return pd.DataFrame(columns=['title', 'genres', 'hybrid_score', 'reason'])
    if cb_recs.empty:
        cf_recs['reason'] = 'Selera Anda Mirip Pengguna Lain'
        return cf_recs.head(top_n).rename(columns={'score': 'hybrid_score'})
    if cf_recs.empty:
        cb_recs['reason'] = 'Mirip Film Pilihan Anda'
        return cb_recs.head(top_n).rename(columns={'score': 'hybrid_score'})

    # 3. Hybrid: Re-ranking
    # Normalisasi skor (Skor CF: 0.5-5, Skor CB: 0-1)
    cb_recs['score_norm'] = cb_recs['score']
    cf_recs['score_norm'] = (cf_recs['score'] - 0.5) / 4.5 # Normalisasi (0-1)
    
    # Gabungkan (Merge) berdasarkan judul
    hybrid_df = pd.merge(cb_recs, cf_recs, on='title', suffixes=('_cb', '_cf'), how='outer').fillna(0)
    
    # Bobot Hybrid: 60% CF (lebih personal), 40% CB (penemuan)
    hybrid_df['hybrid_score'] = (hybrid_df['score_norm_cb'] * 0.4) + (hybrid_df['score_norm_cf'] * 0.6)
    
    # Tentukan alasan
    hybrid_df['reason'] = np.where(
        (hybrid_df['score_norm_cb'] > 0) & (hybrid_df['score_norm_cf'] > 0), 'Gabungan (Mirip Pilihan Anda & Pilihan Pengguna Lain)',
        np.where(hybrid_df['score_norm_cb'] > 0, 'Mirip Film Pilihan Anda', 'Selera Anda Mirip Pengguna Lain')
    )
    
    hybrid_df['genres'] = hybrid_df['genres_cb'].where(hybrid_df['genres_cb'] != 0, hybrid_df['genres_cf'])
    hybrid_df = hybrid_df.sort_values(by='hybrid_score', ascending=False)
    
    return hybrid_df[['title', 'genres', 'hybrid_score', 'reason']].head(top_n)

@st.cache_data
def get_recommendation_explanation(recommended_movie_title, liked_movie_titles, _movies_df, _tfidf_matrix):
    """
    Membuat penjelasan (XAI) mengapa sebuah film direkomendasikan (ON-THE-FLY).
    """
    indices = pd.Series(_movies_df.index, index=_movies_df['title'])
    
    if recommended_movie_title not in indices:
        return "Alasan tidak ditemukan."
        
    rec_idx = indices[recommended_movie_title]
    rec_vector = _tfidf_matrix[rec_idx] # Vektor TF-IDF dari film rekomendasi
    
    best_match_title = None
    best_sim_score = -1.0
    
    for title in liked_movie_titles:
        if title in indices:
            liked_idx = indices[title]
            liked_vector = _tfidf_matrix[liked_idx] # Vektor TF-IDF film yg disukai
            
            # Hitung simularity on-the-fly (1 vs 1)
            sim = cosine_similarity(rec_vector, liked_vector)[0][0]
            
            if sim > best_sim_score:
                best_sim_score = sim
                best_match_title = title
                
    if best_match_title:
        # Dapatkan genre dari kedua film
        rec_genres = set(_movies_df.loc[rec_idx]['genres'].split('|'))
        match_genres = set(_movies_df.loc[indices[best_match_title]]['genres'].split('|'))
        
        # Cari irisan genre
        common_genres = list(rec_genres.intersection(match_genres))
        
        if common_genres:
            return f"Karena Anda menyukai **{best_match_title}**, kami merekomendasikan film ini yang juga memiliki genre: **{', '.join(common_genres)}**."
        else:
            return f"Rekomendasi ini mirip dengan selera pengguna lain yang juga menyukai **{best_match_title}**."
    
    return "Film ini direkomendasikan berdasarkan tren pengguna dengan selera yang mirip dengan Anda."

# --- Memuat Data ---
# Perhatikan ada 5 variabel sekarang
movies_df, ratings_df, svd_model, tfidf_vectorizer, tfidf_matrix = load_models()

# --- UI Streamlit ---
if movies_df is not None:
    st.title("🎬 Sistem Rekomendasi Film Hibrida (Versi Ringan)")
    st.markdown("Dibangun oleh Nadhifa Aqilla Husna - Final Project Data Science Certification")
    
    # Mendapatkan daftar semua film dan user
    all_movie_titles = movies_df['title'].sort_values().tolist()
    all_user_ids = ratings_df['userId'].unique().tolist()

    # --- Area Input ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Pilih User")
        user_id = st.number_input(
            "Masukkan User ID Anda (Contoh: 1, 10, 50):", 
            min_value=min(all_user_ids), 
            max_value=max(all_user_ids),
            value=1
        )

    with col2:
        st.subheader("2. Pilih Film Favorit Anda")
        liked_movie_titles = st.multiselect(
            "Pilih minimal 5 film yang Anda sukai (bisa ketik untuk mencari):",
            all_movie_titles,
            default=[] # Biarkan kosong
        )
    
    st.divider()

    # --- Tombol Generate ---
    if st.button("Dapatkan Rekomendasi Saya!", type="primary", use_container_width=True):
        if user_id and len(liked_movie_titles) >= 5:
            with st.spinner("Mencari film terbaik untuk Anda... 🍿 (Menghitung similaritas on-the-fly)"):
                
                # Mendapatkan rekomendasi
                # Kita passing model/matriks yang sudah di-load
                recommendations = get_hybrid_recommendations(
                    user_id, 
                    liked_movie_titles, 
                    movies_df, 
                    tfidf_matrix, # <-- Perubahan
                    svd_model, 
                    ratings_df,
                    top_n=10
                )
                
                if recommendations.empty:
                    st.warning("Maaf, kami tidak dapat menemukan rekomendasi yang cocok. Coba pilih film lain.")
                else:
                    st.subheader(f"Rekomendasi Teratas untuk User {user_id}:")
                    
                    # Menampilkan hasil
                    for index, row in recommendations.iterrows():
                        st.markdown(f"### {index + 1}. {row['title']}")
                        st.caption(f"Genres: {row['genres']} | Alasan: {row['reason']}")
                        
                        # --- Ini adalah bagian XAI (Explainable AI) ---
                        with st.expander("Mengapa film ini direkomendasikan?"):
                            explanation = get_recommendation_explanation(
                                row['title'],
                                liked_movie_titles,
                                movies_df,
                                tfidf_matrix # <-- Perubahan
                            )
                            st.markdown(explanation)

        elif len(liked_movie_titles) < 5:
            st.error("Silakan pilih minimal 5 film favorit untuk mendapatkan rekomendasi Content-Based yang akurat.")
        else:
            st.error("Silakan masukkan User ID yang valid.")
else:
    st.error("Gagal memuat model. Pastikan pipeline telah dijalankan dengan benar.")

