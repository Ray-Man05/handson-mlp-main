# Learning Machine Learning

Notes and projects as I work through machine learning fundamentals.

## Book Notes

Working through **Hands-On Machine Learning with Scikit-Learn and PyTorch** by Aurélien Géron.

**Completed Chapters:**

- **Chapter 1**: ML fundamentals, definitions and high level overview.
- **Chapter 2**: End-to-end ML project
  - Data exploration and visualization
  - Data preprocessing with imputation and scaling
  - Pipelines and custom transformers
  - Model training and evaluation

## Kaggle Competitions

### House Prices - Advanced Regression Techniques

**Score:** 0.12523

Predicting house sale prices using the Ames Housing dataset.

Built preprocessing pipeline for 79+ features. Trained multiple models with hyperparameter tuning. Used ensemble of best models for final submission.

Built reusable functions for plotting data, preprocessing, comparing model performance and for defining transformers.
Most feature engineering experiments didn't improve performance.
Some of the biggest improvements in performance occured when taking the log of the target, increasing the train/test split, and using custom pipelines for each of the algorithms instead of a single one throughout/

**Approach:**

- Built reusable components and functions to plot and get information on the dataset.
- Defined custom transformers and assembled them into a single reusable pipeline.
- Experimented with feature engineering (logarithm, KMeans clustering, new feature interactions)
- Trained 6+ models with hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- Evaluated using RMSE, plotted performance of best models and compared their predictions
- Final submission uses ensemble predictions combining best performers
