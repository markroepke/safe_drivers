#!/usr/bin/env python3

print("Starting script.")

##########
# Set Up #
##########

print("Importing modules.")

# Import modules
import numpy as np
import pandas as pd
from sklearn import *

#############
# Load Data #
#############

print("Loading data.")

# Load training data
train_val_data = pd.read_csv("../input/train_100.csv")
train_val_target = train_val_data[["target"]]
train_val_data = train_val_data.drop("target", axis = 1)

# Load test data
test_data = pd.read_csv("../input/test_100.csv")

##############
# Clean Data #
##############

print("Cleaning data.")

# Add data key columns
train_val_data = train_val_data.assign(data = "train")
test_data = test_data.assign(data = "test")

# Combine data sets
combined_data = train_val_data.append(test_data)

# Replace -1 values with NaN -- competition info said -1 meant missing
combined_data = combined_data.replace(-1, np.NaN)

# One Hot Encoding
cat_columns = [c for c in list(combined_data.columns.values) if "cat" in c]
for col in cat_columns:
  combined_data[col] = combined_data[col].astype(str)
combined_data = pd.get_dummies(combined_data, columns = cat_columns).drop(cat_columns)

# Create dummy variables for missing values
features = [c for c in list(combined_data.columns.values) if "data" not in c]
combined_data = pd.get_dummies(combined_data, columns = features, dummy_na = True)

##############
# Write Data #
##############

print("Writing data.")

combined_data.to_csv("../input/combined_data.csv")
train_val_target.to_csv("../input/train_val_target.csv")


###
