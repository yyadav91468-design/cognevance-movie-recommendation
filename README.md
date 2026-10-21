# CineMatch — Movie Recommendation System

## Overview

CineMatch is a Python-based movie recommendation system that uses collaborative filtering to recommend movies based on user-rating patterns. It also provides movie popularity analysis, genre insights, and an interactive Streamlit dashboard.

This project was developed as an AI/ML project for the Cognevance Technologies project assignment.

## Features

- MovieLens dataset preprocessing and cleaning
- User-based collaborative filtering
- Personalized movie recommendations
- Popular movie rankings
- Movie-rating and genre visualizations
- Interactive Streamlit dashboard
- CSV export of recommendations
- Evaluation against a popularity-based baseline

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit

## Dataset

This project uses the MovieLens 100K dataset provided by GroupLens Research.

Dataset: https://grouplens.org/datasets/movielens/

Download and extract the `ml-100k` dataset into `data/raw/`. The required files are `u.data` and `u.item`.

Check the dataset's terms before redistributing the original files.

## Project Structure

```text
cognevance_movie_recommendation/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_preprocessing.py
│   ├── recommendation_engine.py
│   └── visualization.py
├── reports/
│   └── figures/
├── screenshots/
├── app.py
├── evaluate.py
├── requirements.txt
├── README.md
└── report.md
```

## Installation

Use Python 3.12 if available.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the Project

First, preprocess the dataset:

```powershell
python -m src.data_preprocessing
```

Generate visualizations:

```powershell
python -m src.visualization
```

Evaluate the recommendation model:

```powershell
python evaluate.py
```

Launch the web application:

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Recommendation Method

The application constructs a user–movie rating matrix and computes cosine similarity between users. It then uses ratings from similar users to rank movies that the target user has not rated.

The popularity-based recommendation baseline ranks movies by their number of training-set ratings.

## Evaluation

The evaluation script holds out a portion of each eligible user's ratings and measures Precision@10 and Recall@10 for relevant held-out movies, defined as ratings of 4 or higher. It compares collaborative filtering with a popularity-based baseline.

Add your actual results from `reports/model_comparison.csv` to the final report.

## Limitations

- New users and movies may have insufficient interaction history.
- User-based collaborative filtering can be affected by sparse rating data.
- Predicted ratings are estimates, not guaranteed preferences.
- The current implementation uses a fixed neighbor count.
- Recommendations depend on the coverage and quality of the dataset.

## Future Improvements

- Add item-based collaborative filtering.
- Tune model parameters using validation data.
- Add hybrid recommendations using movie genres.
- Improve the interface with movie posters.
- Add automated tests and additional recommendation metrics.

## Author

Developed for the Cognevance Technologies AI/ML project assignment.
