import numpy as np
import pandas as pd
import Matplotlib.pyplot as plt
import sys
# Update imports as needed

# Currently set to only take one input file will update as needed
file1 = sys.argv[1]

data = pd.read_csv(file1)

# TODO: Transform data and run statistical tests

data = pd.to_csv("output-analyse.csv")
