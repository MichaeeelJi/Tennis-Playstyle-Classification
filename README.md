# Tennis-Playstyle-Classification
## 1. Tennis Player Style Classification using Gaussian Mixture Models (GMM)

This project aims to classify professional tennis players into different play styles using machine learning, specifically Gaussian Mixture Models (GMM). The project leverages a dataset containing match statistics of ATP players to identify patterns in their playing characteristics and group them into distinct categories.

## Project Overview

The core of this project involves using GMM to cluster players based on their playing styles. The process includes:

1. **Data Preprocessing:** Preparing the dataset by cleaning, handling missing values, and selecting relevant features that capture player characteristics.
2. **GMM Clustering:** Applying Gaussian Mixture Models to the preprocessed data to identify clusters representing different play styles.
3. **Play Style Identification:** Interpreting the clusters to assign play style labels such as 'Counter Puncher', 'Attacking Baseliner', 'All-Court Player', and 'Solid Baseliner'.
4. **Visualization and Analysis:** Visualizing the clusters and analyzing the characteristics of players within each group.

## Key Features

* **Data-driven approach:** The play style classifications are derived from actual match statistics of professional tennis players.
* **Unsupervised learning:** GMM clustering is an unsupervised technique, which means it doesn't require pre-labeled data for training.
* **Interpretable clusters:** The resulting clusters are designed to represent recognizable play styles.
* **Visualization:** The project includes visualizations to aid in understanding the clusters and the distribution of players within them.

## Getting Started

1. **Clone the repository** 
2. **Install dependencies:** Ensure you have the necessary Python libraries installed.
3. **Run the code:** Execute the main script to perform the clustering and analysis.
4. **Explore the results:** Examine the output files and visualizations to understand the play style classifications.

## Dataset

The project utilizes a dataset containing detailed match statistics of ATP players. This dataset is included in the repository.
**feature columns:** firstServeIn_per	secondServeIn_per	ace_per	first_serve_won_per	second_serve_won_per	forehand_winner_per	backhand_winner_per	forehand_unforced_error_per	backhand_unforced_error_per	net_per	net_point_direct_win_per	net_point_winning_per	net_point_error	passing_per	net_point_winByError_per	winner_per	err_per	pts_won_Ite_3_shots_per	shots_in_pts_won_per	shots_in_pts_lost_per	shots_in_won_vs_lost_ratio	inside_in_per	inside_out_per

## Categorization

**the playstyle categories are:** counter puncher, attacking baseliner, solid baseliner, and all-court player. (the GMM category names were initially unknown and need to be manually labeled)
1. Select the top 10 players with the highest score from each category.
2. take the mean of their features
3. make a side-by-side barplot that compares the average feature-wide performance of top 10 players in each category.
4. Web scraping to find out the label for each category by looking at feature-wide performance in each category
5. put the label onto each category in GMM

## Results

The results of the GMM clustering provide insights into the different playing styles present in professional tennis. The project categorizes players into four distinct play styles:

* **Counter Puncher:** Players who excel at retrieving and extending rallies.
* **Attacking Baseliner:** Players who rely on powerful groundstrokes.
* **All-Court Player:** Players with versatile skills and tactical flexibility.
* **Solid Baseliner:** Players who maintain consistent and balanced gameplay.

## Future Work (Tentative) 

* **Model refinement:** Exploring different clustering algorithms and feature engineering techniques to improve the accuracy and interpretability of the play style classifications.
* **Dynamic analysis:** Analyzing player performance over time to identify trends and changes in play styles.
* **Predictive modeling:** Utilizing the play style classifications to predict match outcomes or player performance.

## 2. Random Forest Regressor for Player Style Prediction

This repository contains a machine learning model built using a Random Forest Regressor to predict tennis player styles based on their shot-level statistics.

## Project Overview

The goal of this project is to develop a regression model that can accurately classify tennis players into different play styles (e.g., Counter Puncher, Attacking Baseliner) based on their in-game performance metrics.

## Dataset

The dataset used for this project contains shot-level statistics for tennis players, including features the same as the dataset for GMM but includes the **four defined playstyles:** 


* `Counter Puncher`
* `Attacking Baseliner`
* `All-Court Player`
* `Solid Baseliner`

## Model

A Random Forest Regressor is used for this prediction task. The model is trained on a subset of the data and evaluated on a held-out test set. 

## Evaluation

The model performance is evaluated using Mean Squared Error (MSE) and R-squared (R²) score. Cross-validation techniques, including k-fold cross-validation and leave-one-out cross-validation, are employed to ensure the model's generalizability.



## Dependencies

* Python 3.x
* pandas
* scikit-learn
* joblib
* numpy
* google-colab (if running in Google Colab)

# 3. Final.ipynb

## Description

This project analyzes tennis match data to construct methods to calculate various shot quality metrics and produces a hexagonal playstyle chart for each player. It provides insights into player performance and tactical approaches.

## Requirements

* Python 3.x
* pandas
* numpy
* matplotlib
* plotly
* joblib
* scikit-learn (if you want to retrain the model)

## Installation
Use code with caution
bash pip install pandas numpy matplotlib plotly joblib scikit-learn


## Data

The project uses a CSV file containing detailed information about each shot in a tennis match. 

* **Important features:** The key features used for analysis include pointScore	gameScore	setScore	isPointStart	pointStartTime	isPointEnd	pointEndTime	pointNumber	isBreakPoint	shotInRally	side	serverName	serverFarNear	firstServeIn	firstServeZone	firstServeXCoord	firstServeYCoord	secondServeIn	secondServeZone	secondServeXCoord	secondServeYCoord	isAce	shotContactX	shotContactY	shotDirection	shotFhBh (or ShotType in the old version)	isSlice	isVolley	isOverhead	isApproach	isDropshot	isExcitingPoint	atNetPlayer1	atNetPlayer2	isLob	shotLocationX	shotLocationY	isWinner	isErrorWideR	isErrorWideL	isErrorNet	isErrorLong	clientTeam	Date	Division	Event	lineupPosition	matchDetails	matchVenue	opponentTeam	player1Name	player2Name	player1Hand	player2Hand	Round	Surface	Notes	tiebreakScore	returnerName	shotHitBy	isInsideOut	isInsideIn	isDoubleFault	pointWonBy	lastShotError	serveResult	serveInPlacement	depth


## Models

A pre-trained Random Forest Regressor model (`rf_regressor_model.joblib`) is used to classify player styles into categories like 'Counter Puncher', 'Attacking Baseliner', 'All-Court Player', and 'Solid Baseliner'. The features used by the model are various shot quality metrics.

## Results

The notebook outputs the following:

* **Shot quality metrics:** Various percentages and ratios related to shot types, outcomes, and performance for both players.
* **Player style predictions:** Predicted style categories for both players, in a hexagonal form showing different percentage scores for each category.
