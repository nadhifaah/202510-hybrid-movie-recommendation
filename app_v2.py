import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Cinematch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- FUNGSI CACHE UNTUK MEMUAT MODEL ---
# Menggunakan cache Streamlit agar model tidak di-load ulang setiap kali ada interaksi
@st.cache_resource
def load_models():
    """
    Memuat semua 5 model/aset .pkl dari folder 'models/'.
    """
    base_dir = "models"
    try:
        svd_model = pickle.load(open(os.path.join(base_dir, 'svd_model.pkl'), 'rb'))
        tfidf_vectorizer = pickle.load(open(os.path.join(base_dir, 'tfidf_vectorizer.pkl'), 'rb'))
        tfidf_matrix = pickle.load(open(os.path.join(base_dir, 'tfidf_matrix.pkl'), 'rb'))
        movies_df = pickle.load(open(os.path.join(base_dir, 'movies_df.pkl'), 'rb'))
        ratings_df = pickle.load(open(os.path.join(base_dir, 'ratings_df.pkl'), 'rb'))
        
        # Membuat mapping judul ke index untuk pencarian cepat
        indices_map = pd.Series(movies_df.index, index=movies_df['title'])
        
        return svd_model, tfidf_vectorizer, tfidf_matrix, movies_df, ratings_df, indices_map
    except FileNotFoundError:
        st.error(f"Error: Folder '{base_dir}' tidak ditemukan atau file .pkl hilang. Pastikan Anda sudah menjalankan notebook pemodelan.")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None, None, None, None, None, None

# Memuat semua model ke memori
svd, vectorizer, tfidf_matrix, movies_df, ratings_df, indices_map = load_models()

# --- FUNGSI HELPER ---

def get_recommendation_explanation(liked_movie_title, rec_movie_title, movies_df, indices_map):
    """
    Menghasilkan penjelasan (XAI) mengapa film direkomendasikan.
    """
    try:
        # Tambahkan .get() untuk penanganan error jika judul tidak ada
        liked_idx = indices_map.get(liked_movie_title)
        rec_idx = indices_map.get(rec_movie_title)
        
        if liked_idx is None or rec_idx is None:
            return "Info film tidak ditemukan."

        liked_movie = movies_df.iloc[liked_idx]
        rec_movie = movies_df.iloc[rec_idx]
        
        # Ekstrak fitur-fitur bersih
        liked_genres = set(liked_movie['genres_clean'])
        rec_genres = set(rec_movie['genres_clean'])
        liked_director = liked_movie['director']
        rec_director = rec_movie['director']
        liked_cast = set(liked_movie['cast_clean'])
        rec_cast = set(rec_movie['cast_clean'])
        
        # Cari kesamaan
        common_genres = list(liked_genres.intersection(rec_genres))
        common_cast = list(liked_cast.intersection(rec_cast))
        
        explanations = []
        
        if common_genres:
            explanations.append(f"**Genre Sama:** `{', '.join(common_genres)}`")
        
        if liked_director == rec_director and liked_director != '':
            explanations.append(f"**Sutradara Sama:** `{liked_director}`")
            
        if common_cast:
            explanations.append(f"**Aktor Sama:** `{', '.join(common_cast)}`")
            
        if not explanations:
            return "Film ini memiliki kemiripan plot atau tema."
            
        return "Film ini direkomendasikan karena kesamaan berikut:\n- " + "\n- ".join(explanations)
        
    except Exception as e:
        return f"Tidak dapat menghasilkan penjelasan: {e}"

def get_content_based_recommendations(liked_movie_titles, n_recommendations, movies_df, tfidf_matrix, indices_map):
    """
    Menghasilkan rekomendasi Content-Based secara on-the-fly.
    """
    try:
        # Dapatkan index dari film yang disukai
        liked_indices = [indices_map[title] for title in liked_movie_titles if title in indices_map]
        
        if not liked_indices:
            return pd.DataFrame(columns=['title', 'score_cb'])

        # Ambil vektor TF-IDF untuk film yang disukai
        liked_vectors = tfidf_matrix[liked_indices]
        
        # Hitung skor similaritas rata-rata (on-the-fly)
        # (N_liked_movies, N_features) x (N_features, N_all_movies) -> (N_liked_movies, N_all_movies)
        cosine_sim_matrix = cosine_similarity(liked_vectors, tfidf_matrix)
        
        # Ambil skor rata-rata
        avg_sim_scores = cosine_sim_matrix.mean(axis=0)
        
        # Buat daftar (index_film, skor_similaritas)
        sim_scores_list = list(enumerate(avg_sim_scores))
        
        # Urutkan berdasarkan skor
        sim_scores_list = sorted(sim_scores_list, key=lambda x: x[1], reverse=True)
        
        # Ambil top N (kita ambil N*10 dulu untuk filtering)
        recommendations = []
        for i, score in sim_scores_list:
            movie_title = movies_df.iloc[i]['title']
            if movie_title not in liked_movie_titles:
                recommendations.append({'title': movie_title, 'score_cb': score})
            if len(recommendations) >= (n_recommendations * 10): # Ambil 10x lebih banyak untuk gabungan
                break
                
        return pd.DataFrame(recommendations)
        
    except Exception as e:
        st.error(f"Error di Content-Based: {e}")
        return pd.DataFrame(columns=['title', 'score_cb'])

def get_collaborative_filtering_recommendations(user_id, n_recommendations, movies_df, ratings_df, svd_model):
    """
    Menghasilkan rekomendasi Collaborative Filtering (SVD).
    """
    try:
        # Cari film yang BELUM ditonton user
        watched_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
        all_movie_ids = movies_df['id'].unique()
        to_predict_movie_ids = np.setdiff1d(all_movie_ids, watched_movie_ids)
        
        if len(to_predict_movie_ids) == 0:
            st.warning(f"User {user_id} sudah menonton semua film di dataset!")
            return pd.DataFrame(columns=['title', 'score_cf'])

        # Buat testset (user_id, movie_id, rating_asli_bohongan)
        testset = [[user_id, movie_id, 4.0] for movie_id in to_predict_movie_ids]
        
        # Prediksi rating untuk semua film itu
        predictions = svd_model.test(testset)
        
        # Urutkan prediksi berdasarkan estimasi rating (est)
        predictions.sort(key=lambda x: x.est, reverse=True)
        
        # Ambil Top N
        recommendations = []
        for pred in predictions[:n_recommendations * 10]: # Ambil 10x lebih banyak untuk gabungan
            movie_id = pred.iid
            # Pastikan movie_id ada di movies_df
            movie_title_series = movies_df[movies_df['id'] == movie_id]['title']
            if not movie_title_series.empty:
                movie_title = movie_title_series.values[0]
                recommendations.append({'title': movie_title, 'score_cf': pred.est})
            
        return pd.DataFrame(recommendations)
        
    except Exception as e:
        st.error(f"Error di Collaborative Filtering: {e}")
        return pd.DataFrame(columns=['title', 'score_cf'])


# --- UI STREAMLIT ---

st.title("🎬 Cinematch: Hybrid Movie Recommender System")
st.markdown("""
Dibangun oleh **Nadhifa Aqilla Husna** untuk **Final Project DQ Lab MAI 19** \n
Aplikasi ini menggabungkan dua metode untuk memberikan rekomendasi film yang personal:
1.  **Content-Based:** Menganalisis film yang Anda sukai (berdasarkan plot, genre, aktor, sutradara) untuk mencari film serupa.
2.  **Collaborative Filtering:** Menemukan pengguna lain dengan selera mirip (berdasarkan histori rating) dan merekomendasikan film yang mereka sukai.
""")

# --- Sidebar untuk Input ---
st.sidebar.header("⚙️ Input Pengguna")

# Cek apakah model berhasil di-load
if movies_df is not None:
    # --- Input 1: User ID ---
    user_list = ratings_df['userId'].unique().tolist()
    user_id = st.sidebar.selectbox(
        "Pilih User ID Anda:",
        user_list,
        index=0
    )

    # --- Input 2: Film Favorit (untuk Content-Based) ---
    movie_list = movies_df['title'].sort_values().tolist()
    liked_movies = st.sidebar.multiselect(
        "Pilih 3-5 film favorit Anda (untuk Content-Based):",
        movie_list,
        max_selections=5
    )
    
    # --- Input 3: Jumlah Rekomendasi ---
    n_recs = st.sidebar.slider(
        "Jumlah Rekomendasi:",
        min_value=5,
        max_value=20,
        value=10
    )
    
    # --- Tombol Generate ---
    submit_button = st.sidebar.button("Dapatkan Rekomendasi", type="primary")

else:
    st.sidebar.error("Model tidak dapat dimuat. Aplikasi tidak dapat berjalan.")
    submit_button = False

# --- Logika Utama Saat Tombol Ditekan ---
if submit_button and movies_df is not None:
    if not liked_movies or len(liked_movies) < 1:
        st.warning("Silakan pilih minimal 1 film favorit untuk menjalankan Content-Based.")
    else:
        with st.spinner(f"Mencari {n_recs} rekomendasi hibrida untuk User {user_id}..."):
            
            # --- 1. Content-Based ---
            cb_recs_df = get_content_based_recommendations(liked_movies, n_recs, movies_df, tfidf_matrix, indices_map)
            
            # --- 2. Collaborative Filtering ---
            cf_recs_df = get_collaborative_filtering_recommendations(user_id, n_recs, movies_df, ratings_df, svd)
            
            if cb_recs_df.empty and cf_recs_df.empty:
                st.error("Gagal mendapatkan rekomendasi. Coba input yang berbeda.")
            else:
                # --- 3. Hybrid Logic ---
                st.header(f"🚀 Rekomendasi Hibrida Teratas untuk Anda")
                
                # Normalisasi Skor
                # Skor CB (Cosine Sim) = 0 s/d 1
                # Skor CF (Rating Est) = 0.5 s/d 5.0
                
                if not cb_recs_df.empty:
                    cb_recs_df['score_norm_cb'] = cb_recs_df['score_cb']
                
                if not cf_recs_df.empty:
                    cf_recs_df['score_norm_cf'] = (cf_recs_df['score_cf'] - 0.5) / (5.0 - 0.5)

                # Gabungkan (merge) kedua DataFrame
                if not cb_recs_df.empty and not cf_recs_df.empty:
                    hybrid_recs_df = pd.merge(cb_recs_df, cf_recs_df, on='title', how='outer')
                elif not cb_recs_df.empty:
                    hybrid_recs_df = cb_recs_df
                else:
                    hybrid_recs_df = cf_recs_df
                    
                # Isi NaN dengan 0
                hybrid_recs_df = hybrid_recs_df.fillna(0)
                
                # Hitung Skor Hibrida (Bobot 50/50)
                # Jika salah satu skor 0, skor hibrida hanya akan mengambil dari yang lain
                if 'score_norm_cb' not in hybrid_recs_df: hybrid_recs_df['score_norm_cb'] = 0
                if 'score_norm_cf' not in hybrid_recs_df: hybrid_recs_df['score_norm_cf'] = 0

                hybrid_recs_df['score_hybrid'] = (hybrid_recs_df['score_norm_cb'] * 0.75) + (hybrid_recs_df['score_norm_cf'] * 0.25)
                
                # Urutkan berdasarkan skor hibrida
                hybrid_recs_df = hybrid_recs_df.sort_values('score_hybrid', ascending=False)
                
                # Hapus film yang sudah disukai
                final_recs_df = hybrid_recs_df[~hybrid_recs_df['title'].isin(liked_movies)].head(n_recs)

                # --- Tampilkan Hasil ---
                # Bagi layout jadi 5 kolom
                cols = st.columns(5)
                # Ambil 5 rekomendasi teratas untuk ditampilkan dengan poster
                top_5_recs = final_recs_df.head(5)

                for i, row in enumerate(top_5_recs.itertuples()):
                    with cols[i]:
                        # Gunakan placeholder gambar
                        st.image(f"https://placehold.co/400x600/222/FFF?text={row.title.replace(':', '%3A')}", use_container_width=True)
                        st.markdown(f"**{row.title}**")
                        st.markdown(f"Skor Hibrida: `{row.score_hybrid:.2f}`")
                        
                        # XAI (Explainable AI)
                        with st.expander("Mengapa film ini?"):
                            best_explanation = "Rekomendasi ini didasarkan pada selera pengguna lain yang mirip dengan Anda (Collaborative Filtering)."
                            if row.score_norm_cb > 0:
                                # Jika ada skor CB, cari penjelasan CB
                                explanations = []
                                for liked_movie in liked_movies:
                                    exp = get_recommendation_explanation(liked_movie, row.title, movies_df, indices_map)
                                    if "Sama" in exp: # Jika ada kesamaan spesifik
                                        explanations.append(f"**Karena Anda menyukai '{liked_movie}':**\n{exp}")
                                
                                if explanations:
                                    best_explanation = "\n\n".join(explanations)
                                else:
                                    best_explanation = f"Film ini memiliki kemiripan plot/tema umum dengan film yang Anda sukai."
                            
                            st.markdown(best_explanation)

                # Tampilkan sisa rekomendasi (jika n_recs > 5) sebagai daftar
                if len(final_recs_df) > 5:
                    st.subheader("Rekomendasi Lainnya:")
                    for i, row in enumerate(final_recs_df.iloc[5:].itertuples()):
                        col1, col2 = st.columns([5, 2])
                        with col1:
                            st.markdown(f"**{i+6}. {row.title}**")
                            st.markdown(f"Skor Hibrida: `{row.score_hybrid:.2f}` | Skor CF: `{row.score_norm_cf:.2f}` | Skor CB: `{row.score_norm_cb:.2f}`")
                        with col2:
                            with st.expander("Mengapa film ini?"):
                                best_explanation = "Rekomendasi ini didasarkan pada selera pengguna lain yang mirip dengan Anda (Collaborative Filtering)."
                                if row.score_norm_cb > 0:
                                    explanations = []
                                    for liked_movie in liked_movies:
                                        exp = get_recommendation_explanation(liked_movie, row.title, movies_df, indices_map)
                                        if "Sama" in exp:
                                            explanations.append(f"**Karena Anda menyukai '{liked_movie}':**\n{exp}")
                                    if explanations:
                                        best_explanation = "\n\n".join(explanations)
                                    else:
                                        best_explanation = f"Film ini memiliki kemiripan plot/tema umum dengan film yang Anda sukai."
                                st.markdown(best_explanation)

