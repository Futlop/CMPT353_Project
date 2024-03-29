import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
# Update imports as needed

# Currently set to only take one input file will update as needed
file1 = sys.argv[1]

data = pd.read_csv(file1)

# Separate data into X and y training and validation sets
# X = pd.DataFrame(...)
# y = pd.DataFrame(...)

# X_train, X_valid, y_train, y_valid = train_test_split(X, y)
