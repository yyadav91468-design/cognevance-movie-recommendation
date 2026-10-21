
from pathlib import Path

import pandas as pd
import streamlit as st

from src.recommendation_engine import MovieRecommendationEngine


PROJECT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide"
)


@st.cache_resource
def load_engine():
    return MovieRecommendationEngine()


@st.cache_data
def load_ratings():
    path = PROJECT_DIR / "data" / "processed" / "ratings_clean.csv"
    return pd.read_csv(path)


@st.cache_data
def load_movies():
    path = PROJECT_DIR / "data" / "processed" / "movies_clean.csv"
    return pd.read_csv(path)


st.title("🎬 CineMatch")
st.subheader("Movie Recommendation System")
st.write(
    "Discover movies using collaborative filtering and "
    "explore insights from MovieLens ratings."
)

try:
    engine = load_engine()
    ratings = load_ratings()
    movies = load_movies()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(f"Could not load project data: {error}")
    st.info(
        "Check your MovieLens dataset and run "
        "`python -m src.data_preprocessing` first."
    )
    st.stop()


# Sidebar controls
st.sidebar.header("Recommendation Settings")

available_users = sorted(
    engine.user_movie_matrix.index.astype(int).tolist()
)

user_id = st.sidebar.selectbox(
    "Select a user ID",
    options=available_users,
    index=0
)

number_of_recommendations = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)

number_of_neighbors = st.sidebar.slider(
    "Similar users to consider",
    min_value=5,
    max_value=50,
    value=30
)


# Summary metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Users", f"{ratings['user_id'].nunique():,}")
col2.metric("Movies", f"{movies['movie_id'].nunique():,}")
col3.metric("Ratings", f"{len(ratings):,}")
col4.metric("Average Rating", f"{ratings['rating'].mean():.2f} / 5")

st.divider()


# Main navigation
recommendation_tab, popular_tab, insights_tab = st.tabs(
    ["✨ Recommendations", "🔥 Popular Movies", "📊 Data Insights"]
)


with recommendation_tab:
    st.header(f"Recommended for User {user_id}")

    st.caption(
        "Predictions are based on the ratings of similar users. "
        "They are estimates, not guaranteed personal preferences."
    )

    try:
        recommendations = engine.get_recommendations(
            user_id=int(user_id),
            n_recommendations=number_of_recommendations,
            n_neighbors=number_of_neighbors
        )
    except ValueError as error:
        st.error(str(error))
        recommendations = pd.DataFrame()

    if recommendations.empty:
        st.warning(
            "No recommendations are available for this selection."
        )
    else:
        display_recommendations = recommendations[
            [
                "title",
                "genres",
                "predicted_rating",
                "average_rating",
                "rating_count"
            ]
        ].copy()

        display_recommendations = display_recommendations.rename(
            columns={
                "title": "Movie",
                "genres": "Genres",
                "predicted_rating": "Predicted Rating",
                "average_rating": "Dataset Average",
                "rating_count": "Number of Ratings"
            }
        )

        display_recommendations["Predicted Rating"] = (
            display_recommendations["Predicted Rating"].round(2)
        )
        display_recommendations["Dataset Average"] = (
            display_recommendations["Dataset Average"].round(2)
        )

        st.dataframe(
            display_recommendations,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="Download Recommendations as CSV",
            data=display_recommendations.to_csv(index=False),
            file_name=f"recommendations_user_{user_id}.csv",
            mime="text/csv"
        )


with popular_tab:
    st.header("Most Popular Movies")

    popular_movies = engine.get_popular_movies(limit=20)

    popular_display = popular_movies[
        ["title", "genres", "average_rating", "rating_count"]
    ].rename(
        columns={
            "title": "Movie",
            "genres": "Genres",
            "average_rating": "Average Rating",
            "rating_count": "Number of Ratings"
        }
    )

    st.dataframe(
        popular_display.round({"Average Rating": 2}),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Popular movies have at least 50 ratings in the dataset."
    )


with insights_tab:
    st.header("MovieLens Dataset Insights")

    left, right = st.columns(2)

    with left:
        st.subheader("Rating Distribution")

        rating_counts = (
            ratings["rating"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(rating_counts)

    with right:
        st.subheader("Movies by Genre")

        genre_counts = (
            movies["genres"]
            .fillna("")
            .str.split(", ")
            .explode()
        )

        genre_counts = genre_counts[
            genre_counts.ne("")
            & genre_counts.ne("unknown")
            & genre_counts.ne("Unknown")
        ]

        st.bar_chart(
            genre_counts.value_counts().head(15)
        )

    st.subheader("Generated Visualizations")

    figure_files = [
        ("Rating Distribution", "rating_distribution.png"),
        ("Top-Rated Movies", "top_rated_movies.png"),
        ("Most Popular Movies", "popular_movies.png"),
        ("Movie Genre Distribution", "genre_distribution.png"),
    ]

    for label, filename in figure_files:
        image_path = FIGURES_DIR / filename

        if image_path.exists():
            with st.expander(label):
                st.image(
                    str(image_path),
                    caption=label,
                    use_container_width=True
                )
        else:
            st.info(
                f"{filename} is missing. Run "
                "`python -m src.visualization` to generate it."
            )


st.divider()

st.caption(
    "CineMatch | Built with Python, Pandas, Scikit-learn and Streamlit "
    "| Dataset: MovieLens 100K"
)
