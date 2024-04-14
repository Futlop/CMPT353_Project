# CMPT353_Project

Required libraries:
openpyxl, numpy, pandas, sys, scikit-learn, scipy, matplotlib, datetime, hmmlearn, seaborn, xgboost

How to run the project:
1. Cleaning data
	- Run python data_clean.py on the command line to remove unwanted data
	- Run python add_negatives2.py on the command line to convert dates to datetimes

2. Statistical analysis
	- Run python fire_analyse.py on the command line to determine correlations between variables

3. Machine Learning
	- Run python fire_model.py fire_data.csv on the command line to train a model to predict fire size classes
	- Run python fire_spread_rate_model.py on the command line to train a model to predict fire spread rates
	- Run python hmm.py on the command line to train an HMM to predict fire size classes

4. Data Visualization
	- Run python ml_visualization.py on the command line to visualize fire_model.py results