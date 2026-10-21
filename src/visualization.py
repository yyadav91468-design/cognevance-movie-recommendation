
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"


def load_data():
    ratings_path = DATA_DIR / "ratings_clean.csv"
    movies_path = DATA_DIR / "movies_clean.csv"

    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            "Processed datasets are missing. "
            "Run: python -m src.data_preprocessing"
        )

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    return ratings, movies


def plot_rating_distribution(ratings):
    counts = (
        ratings["rating"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("Distribution of Movie Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Number of Ratings")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "rating_distribution.png",
        dpi=150
    )
    plt.close()


def plot_top_rated_movies(ratings, movies):
    average_ratings = ratings.groupby("movie_id").agg(
        average_rating=("rating", "mean"),
        rating_count=("rating", "count")
    )

    # Require at least 50 ratings for a more reliable comparison.
    qualified = average_ratings[
        average_ratings["rating_count"] >= 50
    ]

    top_movies = (
        qualified
        .merge(movies[["movie_id", "title"]], on="movie_id")
        .nlargest(10, "average_rating")
        .sort_values("average_rating")
    )

    plt.figure(figsize=(11, 7))
    plt.barh(
        top_movies["title"],
        top_movies["average_rating"]
    )

    plt.title("Top 10 Highest-Rated Movies")
    plt.xlabel("Average User Rating")
    plt.ylabel("Movie")
    plt.xlim(0, 5)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "top_rated_movies.png",
        dpi=150
    )
    plt.close()


def plot_popular_movies(ratings, movies):
    popular = (
        ratings.groupby("movie_id")
        .size()
        .reset_index(name="rating_count")
        .merge(movies[["movie_id", "title"]], on="movie_id")
        .nlargest(10, "rating_count")
        .sort_values("rating_count")
    )

    plt.figure(figsize=(11, 7))
    plt.barh(
        popular["title"],
        popular["rating_count"]
    )

    plt.title("Top 10 Most Rated Movies")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Movie")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "popular_movies.png",
        dpi=150
    )
    plt.close()


def plot_genre_distribution(movies):
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

    genre_counts = genre_counts.value_counts().sort_values()

    plt.figure(figsize=(10, 7))
    genre_counts.plot(kind="barh")

    plt.title("Movie Genre Distribution")
    plt.xlabel("Number of Movies")
    plt.ylabel("Genre")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "genre_distribution.png",
        dpi=150
    )
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ratings, movies = load_data()

    plot_rating_distribution(ratings)
    plot_top_rated_movies(ratings, movies)
    plot_popular_movies(ratings, movies)
    plot_genre_distribution(movies)

    print("EDA completed successfully!")
    print(f"Users: {ratings['user_id'].nunique():,}")
    print(f"Ratings: {len(ratings):,}")
    print(f"Movies: {movies['movie_id'].nunique():,}")
    print(f"Overall average rating: {ratings['rating'].mean():.2f}")

    print("\nCharts saved to:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()
