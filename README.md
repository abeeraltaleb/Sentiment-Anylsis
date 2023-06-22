# Sentiment Analysis with NLTK and VADER

This repository contains code for performing sentiment analysis on a dataset of reviews using the Natural Language Toolkit (NLTK) and the VADER (Valence Aware Dictionary and Sentiment Reasoner) sentiment analysis tool.

## Requirements

To run this code, you will need:

- Python 3.x
- pandas
- matplotlib
- seaborn
- nltk
- tqdm

## Usage

1. Clone this repository to your local machine.
2. Download the Reviews.csv dataset from [Kaggle](https://www.kaggle.com/code/robikscube/sentiment-analysis-python-youtube-tutorial/input?select=Reviews.csv) and place it in the same directory as the cloned repository.
3. Open a terminal or command prompt and navigate to the directory containing the repository and the dataset.
4. Run the sentiment_analysis.py script using the command python sentiment_analysis.py.

## Description

The sentiment_analysis.py script performs sentiment analysis on the Reviews.csv dataset using NLTK and VADER. The script reads in the dataset using pandas, plots the count of reviews by stars using matplotlib, and performs sentiment analysis on a single example review using NLTK. The script then runs the VADER polarity score on the entire dataset and merges the results with the original dataset using pandas. Finally, the script plots the compound score, as well as the positive, neutral, and negative scores, by Amazon star review using seaborn.
