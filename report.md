# Movie Recommendation System

## 1. Project Overview

The Movie Recommendation System recommends movies to users based on historical movie ratings. It uses collaborative filtering to identify users with similar rating patterns and compares its performance with a popularity-based recommendation baseline.

## 2. Objectives

- Prepare and clean the MovieLens dataset.
- Analyze movie ratings and genre distributions.
- Develop a personalized movie recommendation algorithm.
- Compare collaborative filtering with a popularity-based baseline.
- Evaluate recommendation quality using Precision@10 and Recall@10.
- Present recommendations and data insights through a Streamlit web application.

## 3. Technologies Used

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Web application:** Streamlit
- **Dataset:** MovieLens 100K

## 4. Methodology

### Data preprocessing

The MovieLens dataset is loaded and cleaned. User ratings and movie metadata are processed and saved as CSV files for reuse.

### Exploratory data analysis

Visualizations are generated to examine rating distributions, popular movies, highly rated movies, and genre distributions.

### Collaborative filtering

A user-item rating matrix is created. Cosine similarity measures the similarity between users. Ratings from similar users are then used to estimate scores for movies that the target user has not rated.

### Popularity baseline

Movies are ranked by the number of ratings and their average rating. Movies already rated by the target user are excluded from recommendations.

### Evaluation

The dataset is divided into training and test sets using a user-wise random split. Ratings of 4.0 or higher are considered relevant. Both approaches are evaluated using Precision@10 and Recall@10.

## 5. Evaluation Results

Insert the metrics from the latest run of `python evaluate.py`.

| Metric | Collaborative Filtering | Popularity Baseline |
|---|---:|---:|
| Precision@10 | To be filled | To be filled |
| Recall@10 | To be filled | To be filled |

Precision@10 measures the fraction of the top ten recommendation positions occupied by relevant movies. Recall@10 measures the fraction of relevant held-out movies recovered among the top ten recommendations.

## 6. Application Features

- Personalized movie recommendations.
- Popular movie recommendations.
- Dataset statistics and visualizations.
- Interactive Streamlit interface.
- Downloadable recommendation data where supported by the application.

## 7. Limitations

- Collaborative filtering may perform poorly for users with few ratings.
- New movies and new users face the cold-start problem.
- Treating unrated movies as zero in the user-item matrix is a simplification.
- A random holdout evaluation does not measure future-time recommendation performance.
- The popularity baseline may outperform collaborative filtering on some datasets and evaluation setups.

## 8. Future Enhancements

- Improve collaborative-filtering predictions and ranking.
- Explore item-based collaborative filtering.
- Add content-based recommendations using movie genres and metadata.
- Evaluate additional ranking metrics.
- Add user registration and persistent preference storage.
- Explore hybrid recommendation methods.

## 9. Conclusion

This project demonstrates an end-to-end movie recommendation workflow, from data preprocessing and exploratory analysis to recommendation generation, model evaluation, and interactive deployment. The comparison with a popularity baseline helps identify areas for further model improvement.