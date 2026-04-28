"""
Streamlit frontend for CineMatch - Movie Recommendation System.
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

st.set_page_config(
    page_title="CineMatch",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0f0f0f; }
    [data-testid="stSidebar"] {
        background-color: #141414;
        border-right: 1px solid #2a2a2a;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span { color: #e0e0e0 !important; }
    .main .block-container {
        background-color: #0f0f0f;
        padding-top: 1.5rem;
    }
    input[type="number"] {
        background-color: #2a2a2a !important;
        color: #e0e0e0 !important;
        border: 1px solid #3a3a3a !important;
        border-radius: 6px !important;
    }
    [data-testid="stNumberInput"] input {
        background-color: #2a2a2a !important;
        color: #e0e0e0 !important;
    }
    [data-testid="stNumberInput"] button {
        background-color: #2a2a2a !important;
        color: #e0e0e0 !important;
        border-color: #3a3a3a !important;
    }
    .stButton > button {
        background-color: #E50914 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        width: 100% !important;
        padding: 0.6rem !important;
    }
    .stButton > button:hover { background-color: #b20710 !important; }
    .stSlider > div > div > div { background-color: #E50914 !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    hr { border-color: #2a2a2a !important; }
</style>
""", unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_recommendations(user_id: int, n: int) -> list:
    try:
        response = requests.post(
            f"{API_URL}/recommendations",
            json={"user_id": user_id, "n": n}
        )
        if response.status_code == 200:
            return response.json()["recommendations"]
        return []
    except Exception as e:
        st.error(f"API error: {e}")
        return []


@st.cache_data(ttl=300)
def get_user_history(user_id: int) -> list:
    try:
        response = requests.get(f"{API_URL}/user/{user_id}/history")
        if response.status_code == 200:
            return response.json()["history"]
        return []
    except Exception:
        return []


@st.cache_data(ttl=3600)
def search_tmdb(title: str) -> dict:
    try:
        clean_title = title.split("(")[0].strip()
        response = requests.get(
            f"{TMDB_BASE_URL}/search/movie",
            params={
                "api_key": TMDB_API_KEY,
                "query": clean_title,
                "language": "en-US",
                "page": 1
            }
        )
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                return results[0]
    except Exception:
        pass
    return {}


@st.cache_data(ttl=86400)
def get_tmdb_genres() -> dict:
    try:
        response = requests.get(
            f"{TMDB_BASE_URL}/genre/movie/list",
            params={"api_key": TMDB_API_KEY, "language": "en-US"}
        )
        if response.status_code == 200:
            genres = response.json().get("genres", [])
            return {g["id"]: g["name"] for g in genres}
    except Exception:
        pass
    return {}


def get_poster_url(tmdb_data: dict) -> str:
    poster_path = tmdb_data.get("poster_path")
    if poster_path:
        return f"{TMDB_IMAGE_URL}{poster_path}"
    return None


def get_genres(tmdb_data: dict, max_genres: int = 3) -> str:
    genre_ids = tmdb_data.get("genre_ids", [])
    genres = [tmdb_genres.get(gid, "") for gid in genre_ids[:max_genres]]
    return " · ".join([g for g in genres if g])


# load genres once
tmdb_genres = get_tmdb_genres()


# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.5rem 0 1rem;'>
        <p style='font-size:2.5rem;margin:0;'>🍿</p>
        <p style='font-size:1.6rem;font-weight:700;color:#E50914;margin:0;letter-spacing:-0.5px;'>CineMatch</p>
        <p style='font-size:0.7rem;color:#555;margin:4px 0 0;'>discover what to watch next</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<p style='font-size:0.7rem;color:#888;text-transform:uppercase;letter-spacing:0.06em;margin:0 0 6px;'>User ID</p>", unsafe_allow_html=True)
    user_id = st.number_input(
        "user_id",
        min_value=1,
        max_value=6040,
        value=1,
        step=1,
        label_visibility="collapsed"
    )
    st.markdown("<p style='font-size:0.7rem;color:#555;margin:0 0 16px;'>Valid range: 1 — 6040</p>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.7rem;color:#888;text-transform:uppercase;letter-spacing:0.06em;margin:0 0 6px;'>Recommendations</p>", unsafe_allow_html=True)
    n = st.select_slider(
        "Recommendations",
        options=[5, 10, 20],
        value=10,
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    get_recs = st.button("🍿 Get Recommendations")

    st.markdown("---")

    st.markdown("<p style='font-size:0.7rem;color:#888;text-transform:uppercase;letter-spacing:0.06em;margin:0 0 12px;'>About</p>", unsafe_allow_html=True)

    stats = [
        ("Algorithm", "SVD", "#e0e0e0"),
        ("RMSE", "0.965", "#4CAF50"),
        ("Dataset", "MovieLens 1M", "#e0e0e0"),
        ("Users", "6,040", "#e0e0e0"),
        ("Ratings", "1M+", "#e0e0e0"),
    ]
    for label, value, color in stats:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;margin-bottom:8px;'>
            <span style='font-size:0.75rem;color:#555;'>{label}</span>
            <span style='font-size:0.75rem;color:{color};font-weight:600;'>{value}</span>
        </div>
        """, unsafe_allow_html=True)


# ── main content ─────────────────────────────────────────────────────────────

if get_recs:

    # hero banner
    st.markdown(f"""
    <div style='background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:1.25rem 1.75rem;margin-bottom:1.5rem;display:flex;justify-content:space-between;align-items:center;'>
        <div>
            <p style='font-size:1.3rem;font-weight:700;color:#e0e0e0;margin:0 0 4px;'>User {user_id} 👋</p>
            <p style='font-size:0.8rem;color:#555;margin:0;'>Powered by SVD collaborative filtering · MovieLens 1M · RMSE 0.965</p>
        </div>
        <div style='text-align:right;'>
            <p style='font-size:2rem;font-weight:700;color:#E50914;margin:0;line-height:1;'>{n}</p>
            <p style='font-size:0.7rem;color:#555;margin:0;'>recommendations</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading..."):
        recommendations = get_recommendations(user_id, n)
        history = get_user_history(user_id)

    # side by side layout
    left, right = st.columns(2)

    # LEFT — watch history
    with left:
        st.markdown("""
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:1rem;'>
            <div style='width:3px;height:18px;background:#FFA726;'></div>
            <p style='font-size:1rem;font-weight:600;color:#e0e0e0;margin:0;'>What you watched</p>
        </div>
        """, unsafe_allow_html=True)

        if history:
            for movie in history[:5]:
                tmdb_data = search_tmdb(movie["title"])
                poster_url = get_poster_url(tmdb_data)

                c1, c2, c3 = st.columns([1, 6, 2])
                with c1:
                    if poster_url:
                        st.image(poster_url, width=50)
                    else:
                        st.markdown("<div style='width:50px;height:70px;background:#1a1a1a;border-radius:4px;border:1px solid #2a2a2a;'></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<p style='font-size:0.82rem;font-weight:600;color:#e0e0e0;margin:0 0 2px;'>{movie['title']}</p>", unsafe_allow_html=True)
                    genres = get_genres(tmdb_data)
                    if genres:
                        st.markdown(f"<p style='font-size:0.68rem;color:#E50914;margin:0 0 2px;font-weight:500;'>{genres}</p>", unsafe_allow_html=True)
                    overview = tmdb_data.get("overview", "")[:80] + "..." if tmdb_data.get("overview") else ""
                    if overview:
                        st.markdown(f"<p style='font-size:0.68rem;color:#555;margin:0;'>{overview}</p>", unsafe_allow_html=True)
                with c3:
                    rating = movie.get("rating", "N/A")
                    color = "#4CAF50" if float(rating) >= 4 else "#FFA726" if float(rating) >= 3 else "#EF5350"
                    st.markdown(f"<p style='text-align:right;font-size:0.9rem;font-weight:700;color:{color};margin:0;'>{rating}/5</p>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align:right;font-size:0.65rem;color:#555;margin:0;'>your rating</p>", unsafe_allow_html=True)

                st.markdown("<hr>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#555;font-size:0.85rem;'>No history available.</p>", unsafe_allow_html=True)

    # RIGHT — recommendations
    with right:
        st.markdown("""
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:1rem;'>
            <div style='width:3px;height:18px;background:#E50914;'></div>
            <p style='font-size:1rem;font-weight:600;color:#e0e0e0;margin:0;'>Recommended for you</p>
        </div>
        """, unsafe_allow_html=True)

        if recommendations:
            for movie in recommendations[:5]:
                tmdb_data = search_tmdb(movie["title"])
                poster_url = get_poster_url(tmdb_data)

                c1, c2, c3 = st.columns([1, 6, 2])
                with c1:
                    if poster_url:
                        st.image(poster_url, width=50)
                    else:
                        st.markdown("<div style='width:50px;height:70px;background:#1a1a1a;border-radius:4px;border:1px solid #2a2a2a;'></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<p style='font-size:0.82rem;font-weight:600;color:#e0e0e0;margin:0 0 2px;'>{movie['title']}</p>", unsafe_allow_html=True)
                    genres = get_genres(tmdb_data)
                    if genres:
                        st.markdown(f"<p style='font-size:0.68rem;color:#E50914;margin:0 0 2px;font-weight:500;'>{genres}</p>", unsafe_allow_html=True)
                    overview = tmdb_data.get("overview", "")[:80] + "..." if tmdb_data.get("overview") else ""
                    if overview:
                        st.markdown(f"<p style='font-size:0.68rem;color:#555;margin:0;'>{overview}</p>", unsafe_allow_html=True)
                with c3:
                    score = movie.get("predicted_score", "N/A")
                    st.markdown(f"<p style='text-align:right;font-size:0.9rem;font-weight:700;color:#4CAF50;margin:0;'>{score}/5</p>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align:right;font-size:0.65rem;color:#555;margin:0;'>predicted</p>", unsafe_allow_html=True)

                st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("---")

    # full recommendations grid
    st.markdown("""
    <div style='display:flex;align-items:center;gap:8px;margin-bottom:1rem;'>
        <div style='width:3px;height:18px;background:#E50914;'></div>
        <p style='font-size:1rem;font-weight:600;color:#e0e0e0;margin:0;'>All recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    if recommendations:
        cols = st.columns(5)
        for i, movie in enumerate(recommendations):
            with cols[i % 5]:
                tmdb_data = search_tmdb(movie["title"])
                poster_url = get_poster_url(tmdb_data)

                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.markdown(
                        "<div style='aspect-ratio:2/3;background:#1a1a1a;border-radius:8px;border:1px solid #2a2a2a;display:flex;align-items:center;justify-content:center;'><span style='color:#444;font-size:11px;'>No poster</span></div>",
                        unsafe_allow_html=True
                    )

                st.markdown(f"<p style='font-size:0.72rem;font-weight:600;color:#e0e0e0;margin:4px 0 2px;line-height:1.3;'>{movie['title']}</p>", unsafe_allow_html=True)
                genres = get_genres(tmdb_data)
                if genres:
                    st.markdown(f"<p style='font-size:0.65rem;color:#E50914;margin:0 0 1px;'>{genres}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:0.72rem;color:#4CAF50;font-weight:600;margin:0 0 1px;'>{movie['predicted_score']} / 5</p>", unsafe_allow_html=True)
                tmdb_rating = tmdb_data.get("vote_average")
                if tmdb_rating:
                    st.markdown(f"<p style='font-size:0.68rem;color:#555;margin:0;'>TMDB {tmdb_rating:.1f}</p>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align:center;padding:6rem 2rem;'>
        <p style='font-size:4rem;margin:0 0 1rem;'>🍿</p>
        <p style='font-size:1.8rem;font-weight:700;color:#e0e0e0;margin:0 0 8px;'>Welcome to CineMatch</p>
        <p style='font-size:0.9rem;color:#555;margin:0 0 2rem;'>Your personal AI-powered movie discovery engine</p>
        <div style='display:inline-flex;gap:2rem;background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:1rem 2rem;'>
            <div style='text-align:center;'>
                <p style='font-size:1.3rem;font-weight:700;color:#E50914;margin:0;'>1M+</p>
                <p style='font-size:0.7rem;color:#555;margin:0;'>ratings</p>
            </div>
            <div style='width:1px;background:#2a2a2a;'></div>
            <div style='text-align:center;'>
                <p style='font-size:1.3rem;font-weight:700;color:#E50914;margin:0;'>6,040</p>
                <p style='font-size:0.7rem;color:#555;margin:0;'>users</p>
            </div>
            <div style='width:1px;background:#2a2a2a;'></div>
            <div style='text-align:center;'>
                <p style='font-size:1.3rem;font-weight:700;color:#E50914;margin:0;'>0.965</p>
                <p style='font-size:0.7rem;color:#555;margin:0;'>RMSE</p>
            </div>
            <div style='width:1px;background:#2a2a2a;'></div>
            <div style='text-align:center;'>
                <p style='font-size:1.3rem;font-weight:700;color:#E50914;margin:0;'>SVD</p>
                <p style='font-size:0.7rem;color:#555;margin:0;'>model</p>
            </div>
        </div>
        <p style='font-size:0.8rem;color:#444;margin:2rem 0 0;'>← Enter your User ID in the sidebar to get started</p>
    </div>
    """, unsafe_allow_html=True)