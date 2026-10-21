
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"


class MovieRecommendationEngine:
    """Recommend movies using user-based collaborative filtering."""

    def __init__(self):
        ratings_path = DATA_DIR / "ratings_clean.csv"
        movies_path = DATA_DIR / "movies_clean.csv"

        if not ratings_path.exists() or not movies_path.exists():
            raise FileNotFoundError(
                "Processed datasets not found. Run "
                "'python -m src.data_preprocessing' first."
            )

        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)

        # Each row represents a user; each column represents a movie.
        self.user_movie_matrix = self.ratings.pivot_table(
            index="user_id",
            columns="movie_id",
            values="rating",
            fill_value=0
        )

        # Calculate user-to-user cosine similarity.
        similarity_values = cosine_similarity(
            self.user_movie_matrix
        )

        self.user_similarity = pd.DataFrame(
            similarity_values,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index
        )

        # Precompute movie statistics for reporting.
        self.movie_stats = self.ratings.groupby("movie_id").agg(
            average_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )

    def get_recommendations(
        self,
        user_id: int,
        n_recommendations: int = 10,
        n_neighbors: int = 30
    ) -> pd.DataFrame:
        """Return movies recommended for an existing user."""

        if user_id not in self.user_movie_matrix.index:
            raise ValueError(
                f"User ID {user_id} does not exist in the dataset."
            )

        if n_recommendations < 1 or n_neighbors < 1:
            raise ValueError(
                "Recommendation count and neighbor count must be positive."
            )

        user_ratings = self.user_movie_matrix.loc[user_id]

        # Exclude the target user from the neighbor list.
        similar_users = self.user_similarity.loc[user_id].drop(
            index=user_id
        ).sort_values(ascending=False)

        neighbors = similar_users.head(n_neighbors)

        # Use only movies not yet rated by the target user.
        unseen_movie_ids = user_ratings[
            user_ratings == 0
        ].index

        # Ratings supplied by the neighboring users.
        neighbor_ratings = self.user_movie_matrix.loc[
            neighbors.index, unseen_movie_ids
        ]

        # Weighted ratings: more similar users contribute more.
        weights = neighbors.clip(lower=0)

        weighted_scores = neighbor_ratings.mul(
            weights, axis=0
        ).sum(axis=0)

        # Normalize by the total similarity weight of users
        # who actually rated each candidate movie.
        rated_mask = neighbor_ratings.gt(0)
        effective_weights = rated_mask.mul(weights, axis=0)
        weight_sums = effective_weights.sum(axis=0)

        predicted_scores = weighted_scores.div(
            weight_sums.replace(0, float("nan"))
        )

        recommendations = (
            predicted_scores.dropna()
            .rename("predicted_rating")
            .reset_index()
        )

        recommendations.columns = [
            "movie_id",
            "predicted_rating"
        ]

        recommendations = recommendations.merge(
            self.movies[["movie_id", "title", "genres"]],
            on="movie_id",
            how="left"
        )

        recommendations = recommendations.merge(
            self.movie_stats.reset_index(),
            on="movie_id",
            how="left"
        )

        recommendations = recommendations.sort_values(
            by=["predicted_rating", "rating_count"],
            ascending=[False, False]
        )

        return recommendations.head(n_recommendations).reset_index(
            drop=True
        )

    def get_popular_movies(self, limit: int = 10) -> pd.DataFrame:
        """Return popular movies with at least 50 ratings."""

        popular = self.movie_stats[
            self.movie_stats["rating_count"] >= 50
        ].copy()

        popular = popular.reset_index().merge(
            self.movies[["movie_id", "title", "genres"]],
            on="movie_id",
            how="left"
        )

        popular = popular.sort_values(
            by=["rating_count", "average_rating"],
            ascending=[False, False]
        )

        return popular.head(limit).reset_index(drop=True)


def main():
    engine = MovieRecommendationEngine()

    print("=" * 70)
    print("MOVIE RECOMMENDATION SYSTEM")
    print("=" * 70)

    print(f"\nTotal users: {len(engine.user_movie_matrix):,}")
    print(f"Total movies: {len(engine.movies):,}")

    # User 1 is used as a demonstration; users can be changed.
    user_id = 1

    recommendations = engine.get_recommendations(
        user_id=user_id,
        n_recommendations=10
    )

    print(f"\nTop recommendations for User {user_id}:\n")

    if recommendations.empty:
        print("No recommendations found for this user.")
    else:
        print(
            recommendations[
                [
                    "title",
                    "genres",
                    "predicted_rating",
                    "rating_count"
                ]
            ].to_string(
                index=False,
                formatters={
                    "predicted_rating": lambda value: f"{value:.2f}"
                }
            )
        )

    print("\nTop popular movies:\n")

    print(
        engine.get_popular_movies(10)[
            ["title", "average_rating", "rating_count"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
