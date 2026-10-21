from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "reports"

K = 10
RELEVANT_RATING = 4.0
RANDOM_STATE = 42
N_NEIGHBORS = 30


def load_ratings():
    """Load cleaned MovieLens ratings."""
    path = DATA_DIR / "ratings_clean.csv"

    if not path.exists():
        raise FileNotFoundError(
            "Clean ratings not found. Run "
            "'python -m src.data_preprocessing' first."
        )

    ratings = pd.read_csv(path)

    required_columns = {"user_id", "movie_id", "rating"}
    missing = required_columns - set(ratings.columns)

    if missing:
        raise ValueError(
            f"Ratings dataset is missing columns: {sorted(missing)}"
        )

    return ratings


def split_by_user(ratings):
    """Hold out approximately 20% of each user's ratings."""
    train_parts = []
    test_parts = []

    for _, group in ratings.groupby("user_id"):
        group = group.sample(
            frac=1,
            random_state=RANDOM_STATE
        )

        if len(group) < 5:
            train_parts.append(group)
            continue

        test_size = max(1, round(len(group) * 0.2))
        test_size = min(test_size, len(group) - 1)

        test_parts.append(group.iloc[:test_size])
        train_parts.append(group.iloc[test_size:])

    train = pd.concat(train_parts, ignore_index=True)

    if test_parts:
        test = pd.concat(test_parts, ignore_index=True)
    else:
        test = pd.DataFrame(columns=ratings.columns)

    return train, test


def build_recommender(train):
    """Build the user-item matrix, similarity matrix, and popularity ranking."""
    matrix = train.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        fill_value=0
    )

    similarities = cosine_similarity(matrix)

    similarity_df = pd.DataFrame(
        similarities,
        index=matrix.index,
        columns=matrix.index
    )

    popularity = (
        train.groupby("movie_id")
        .agg(
            rating_count=("rating", "count"),
            average_rating=("rating", "mean")
        )
        .sort_values(
            ["rating_count", "average_rating"],
            ascending=False
        )
    )

    return matrix, similarity_df, popularity


def recommend_collaborative(
    user_id,
    matrix,
    similarities,
    n=K,
    neighbors=N_NEIGHBORS
):
    """Recommend unseen movies using positively similar users."""
    if user_id not in matrix.index:
        return []

    user_ratings = matrix.loc[user_id]

    similar_users = (
        similarities.loc[user_id]
        .drop(index=user_id)
        .sort_values(ascending=False)
    )

    # Ignore users with zero or negative similarity.
    similar_users = similar_users[similar_users > 0].head(neighbors)

    if similar_users.empty:
        return []

    # Candidates are movies absent from the user's training history.
    unseen = user_ratings[user_ratings == 0].index

    if len(unseen) == 0:
        return []

    neighbor_ratings = matrix.loc[similar_users.index, unseen]
    weights = similar_users

    # Only users who actually rated a candidate contribute to its score.
    rated_mask = neighbor_ratings.gt(0)
    weighted_ratings = neighbor_ratings.mul(weights, axis=0)

    numerator = weighted_ratings.sum(axis=0)
    denominator = rated_mask.mul(weights, axis=0).sum(axis=0)

    scores = (
        numerator
        .div(denominator.replace(0, float("nan")))
        .dropna()
    )

    return scores.sort_values(ascending=False).head(n).index.tolist()


def recommend_popular(user_id, train, popularity, n=K):
    """Recommend popular movies the user has not rated in training."""
    seen = set(
        train.loc[train["user_id"] == user_id, "movie_id"]
    )

    candidates = popularity.loc[
        ~popularity.index.isin(seen)
    ]

    return candidates.head(n).index.tolist()


def precision_recall_at_k(recommendations, relevant, k=K):
    """Calculate Precision@K and Recall@K."""
    recommendations = recommendations[:k]

    if not relevant:
        return 0.0, 0.0

    hits = len(set(recommendations) & relevant)

    precision = hits / k
    recall = hits / len(relevant)

    return precision, recall


def evaluate_model(ratings, train, test):
    """Evaluate collaborative filtering against a popularity baseline."""
    matrix, similarities, popularity = build_recommender(train)

    relevant_by_user = (
        test[test["rating"] >= RELEVANT_RATING]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    evaluated_users = [
        user_id
        for user_id, relevant in relevant_by_user.items()
        if relevant and user_id in matrix.index
    ]

    if not evaluated_users:
        raise RuntimeError(
            "No eligible users found for evaluation."
        )

    results = []

    for user_id in evaluated_users:
        relevant = relevant_by_user[user_id]

        collaborative = recommend_collaborative(
            user_id,
            matrix,
            similarities
        )

        popular = recommend_popular(
            user_id,
            train,
            popularity
        )

        cf_precision, cf_recall = precision_recall_at_k(
            collaborative, relevant
        )

        pop_precision, pop_recall = precision_recall_at_k(
            popular, relevant
        )

        results.append({
            "user_id": user_id,
            "relevant_movies": len(relevant),
            "cf_precision_at_10": cf_precision,
            "cf_recall_at_10": cf_recall,
            "pop_precision_at_10": pop_precision,
            "pop_recall_at_10": pop_recall,
        })

    return pd.DataFrame(results), len(evaluated_users)


def main():
    ratings = load_ratings()
    train, test = split_by_user(ratings)

    results_df, users_evaluated = evaluate_model(
        ratings, train, test
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Detailed results for each user.
    results_df.to_csv(
        REPORT_DIR / "evaluation_per_user.csv",
        index=False
    )

    # Explicit metric column prevents the missing-header warning.
    metric_columns = [
        "cf_precision_at_10",
        "cf_recall_at_10",
        "pop_precision_at_10",
        "pop_recall_at_10",
    ]

    summary = (
        results_df[metric_columns]
        .mean()
        .rename_axis("metric")
        .reset_index(name="mean_score")
    )

    summary.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    print("=" * 60)
    print("MOVIE RECOMMENDATION MODEL EVALUATION")
    print("=" * 60)
    print(f"Total ratings: {len(ratings):,}")
    print(f"Training ratings: {len(train):,}")
    print(f"Test ratings: {len(test):,}")
    print(f"Users evaluated: {users_evaluated:,}")
    print(f"Relevant rating threshold: {RELEVANT_RATING:g}")
    print(f"Recommendation cutoff: {K}")
    print(f"Collaborative-filtering neighbors: {N_NEIGHBORS}")
    print("\nMean evaluation scores:")
    print(
        summary.to_string(
            index=False,
            formatters={"mean_score": lambda x: f"{x:.4f}"}
        )
    )
    print(f"\nReports saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()