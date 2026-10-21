
from pathlib import Path

import pandas as pd


# Project directory paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

RATINGS_FILE = RAW_DIR / "u.data"
MOVIES_FILE = RAW_DIR / "u.item"


def load_ratings():
    """Load the MovieLens ratings dataset."""
    if not RATINGS_FILE.exists():
        raise FileNotFoundError(
            f"Ratings file not found: {RATINGS_FILE}"
        )

    ratings = pd.read_csv(
        RATINGS_FILE,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="utf-8"
    )

    return ratings


def load_movies():
    """Load movie titles and genre information."""
    if not MOVIES_FILE.exists():
        raise FileNotFoundError(
            f"Movie file not found: {MOVIES_FILE}"
        )

    genre_columns = [
        "unknown", "Action", "Adventure", "Animation",
        "Children's", "Comedy", "Crime", "Documentary",
        "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    ]

    movie_columns = [
        "movie_id", "title", "release_date", "video_release_date",
        "imdb_url"
    ] + genre_columns

    movies = pd.read_csv(
        MOVIES_FILE,
        sep="|",
        names=movie_columns,
        encoding="latin-1"
    )

    # Convert the binary genre columns into a readable list.
    movies["genres"] = movies[genre_columns].apply(
        lambda row: ", ".join(
            genre for genre in genre_columns
            if row[genre] == 1
        ),
        axis=1
    )

    movies = movies[
        ["movie_id", "title", "release_date", "genres"]
    ]

    return movies


def clean_data(ratings, movies):
    """Validate and clean ratings and movie metadata."""

    # Remove duplicate records.
    ratings = ratings.drop_duplicates(
        subset=["user_id", "movie_id"],
        keep="last"
    ).copy()

    # Convert rating columns to numeric values.
    ratings["user_id"] = pd.to_numeric(
        ratings["user_id"], errors="coerce"
    )
    ratings["movie_id"] = pd.to_numeric(
        ratings["movie_id"], errors="coerce"
    )
    ratings["rating"] = pd.to_numeric(
        ratings["rating"], errors="coerce"
    )

    # Remove missing identifiers and ratings outside 1–5.
    ratings = ratings.dropna(
        subset=["user_id", "movie_id", "rating"]
    )

    ratings = ratings[
        ratings["rating"].between(1, 5)
    ].copy()

    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["movie_id"] = ratings["movie_id"].astype(int)

    # Keep only ratings associated with known movies.
    ratings = ratings[
        ratings["movie_id"].isin(movies["movie_id"])
    ].copy()

    # Remove duplicate movie IDs from the metadata.
    movies = movies.drop_duplicates(
        subset=["movie_id"]
    ).copy()

    # Fill missing titles and genres safely.
    movies["title"] = movies["title"].fillna("Unknown Movie")
    movies["genres"] = movies["genres"].fillna("Unknown")

    return ratings, movies


def main():
    """Run the complete preprocessing pipeline."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MovieLens dataset...")

    ratings = load_ratings()
    movies = load_movies()

    print(f"Original ratings rows: {len(ratings):,}")
    print(f"Original movies rows: {len(movies):,}")

    ratings, movies = clean_data(ratings, movies)

    # Save processed datasets.
    ratings.to_csv(
        PROCESSED_DIR / "ratings_clean.csv",
        index=False
    )

    movies.to_csv(
        PROCESSED_DIR / "movies_clean.csv",
        index=False
    )

    # Save a combined dataset for analysis and the app.
    movie_ratings = ratings.merge(
        movies,
        on="movie_id",
        how="inner"
    )

    movie_ratings.to_csv(
        PROCESSED_DIR / "movie_ratings_clean.csv",
        index=False
    )

    print("\nPreprocessing completed successfully!")
    print(f"Clean ratings: {len(ratings):,}")
    print(f"Unique users: {ratings['user_id'].nunique():,}")
    print(f"Unique rated movies: {ratings['movie_id'].nunique():,}")
    print(f"Movie metadata records: {len(movies):,}")

    print("\nProcessed files saved in:")
    print(PROCESSED_DIR)


if __name__ == "__main__":
    main()
