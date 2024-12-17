# DSCI 553 - Foundations and Applications of Data Mining - Fall 2024

## Introduction

The purpose of this repository is to showcase skills I have gained over the course of taking DSCI 553. 

> [!WARNING]
> Please do not copy the contents of this repository for your own coursework! You may look over my implementation and try to understand the logic, but please write your own code for submission.

### Technologies used: 
* Python
    * PySpark
    * NumPy
    * XGBoost
    * Scikit-Learn
    * GraphFrames

## Homework Scores
| Homework | Topic | Score |
| -------- | ----- | ----- |
| HW1 | MapReduce | 5/7 |
| HW2 | Frequent Itemsets, SON Algorithm | 7/7 |
| HW3 | Locality-Sensitive Hashing and Recommender Systems | 7/7 |
| HW4 | Community Detection, Girvan-Newman Algorithm | 7/7 |
| HW5 | Data Streams, Bloom Filter/Martin-Flajolet Algorithm | 7/7 |
| HW6 | Clustering, Bradley-Fayyad-Reina (BFR) Algorithm | 7/7 |

## Exam Score
* Comprehensive Exam: 30.3/40 (75.75%)

> [!NOTE]
> The class average on the comprehensive exam was a 64%. My score fell into the 80th percentile when compared to my peers' scores.

## Competition Project
* **Goal:** Given a user-business pair on Yelp, accurately predict the rating given to the business by the user.
    * Training dataset of 455,854 points
    * Validation dataset of 142,044 points
* Weighted hybrid recommender system with model-based and item-based collaborative filtering components
* Final RMSE: 0.98477 stars
* Execution time: 96 seconds
* Error distribution:

| Error range (stars) | Number of observations | Percentage of validation set |
| ------------ | ------------ | ---------- |
| \>=0 and <=1 | 101,530 | 71.48% |
| \>1 and <=2 | 33,427 | 23.53% |
| \>2 and <=3 | 6,327 | 4.45% |
| \>3 and <=4 | 760 | 0.54% |
| \>4 and <=5 | 0 | 0.00% |

### Future improvements:
* Including even more features from dataset: number of Yelp friends a user has, compliments on Yelp profile, etc.
* Upgrading existing packages (XGBoost, PySpark, Scikit-Learn) to take advantage of latest features
    * Fitting XGBoost model with `reg:squarederror` loss function, learning rate scheduler, dynamic early stopping threshold
    * Dynamically weighting hybrid recommender system or focusing on features to determine weights
