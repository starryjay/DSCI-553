# DSCI 553 - Foundations and Applications of Data Mining - Fall 2024

The purpose of this repository is to showcase skills I've gained over the course of this semester. 
Please do not use the contents of this repository as your own work for the course.

Technologies used: 
* Python
 * PySpark
 * NumPy
 * XGBoost
 * Scikit-Learn
 * GraphFrames

## Homework Scores
* HW1 - 5/7
* HW2 - 7/7
* HW3 - 7/7
* HW4 - 7/7
* HW5 - 7/7
* HW6 - 7/7

## Exam Score
* Comprehensive Exam: 30.3/40 (75.75%)
* Class average: 64%
* 80th percentile

## Competition Project
* Weighted hybrid recommender system with model-based and item-based collaborative filtering components
* Training dataset of 450,000 points
* Validation dataset of 142,044 points
* Sadly did not pass the 0.9800 RMSE threshold for full credit
* Final RMSE: 0.98477 stars
* Execution time: 96 seconds
* Error Distribution (stars):
  * \>=0 and <=1: 101530
  * \>1 and <=2: 33427
  * \>2 and <=3: 6327
  * \>3 and <=4: 760
  * \>4 and <=5: 0
* Future improvements:
  * Including even more features from dataset: number of Yelp friends a user has, compliments on Yelp profile, etc.
  * Upgrading existing packages (XGBoost, PySpark, Scikit-Learn) to take advantage of latest features
    * Fitting XGBoost model with `reg:squarederror` loss function, learning rate scheduler, dynamic early stopping threshold
    * Dynamically weighting hybrid recommender system or focusing on features to determine weights
