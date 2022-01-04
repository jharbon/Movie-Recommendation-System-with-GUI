# -*- coding: utf-8 -*-

""" This script can be used for selecting subsets of a given training set
and its associated test set based on the minimum number of samples,
seen and unseen, each user ID can have."""

import pandas as pd

from os import chdir
from os.path import dirname, abspath, join

# Set working directory to folder which contains this script
chdir(dirname(abspath(__file__)))

def select_subset(train_path, test_path, min_samples, save_folder):
    # Load training and test sets into dataframes
    train_set = pd.read_csv(train_path)
    test_set = pd.read_csv(test_path)
    
    print(train_set.shape[0])
    print(test_set.shape[0])
    
    # Create boolean index to select user ID values based on minimum sample number condition
    bool_index = train_set.groupby("user_ID")["movie_ID"].transform("count") >= min_samples
    train_set = train_set.loc[bool_index]
    
    # Save new training set in a new folder
    train_set.to_csv(join(save_folder, "train_set.csv"))
    
    # Create series of unique user IDs from new training set and name it "user_ID"
    train_set_IDs = pd.Series(train_set["user_ID"].unique()).rename("user_ID")
                     
    # Join training set user ID column with test set on user ID
    # New test set will contain only user IDs which are in the new training set
    test_set = pd.merge(left = test_set, right = train_set_IDs,
                        on = "user_ID", how = "inner")
    
    print("Train Users " + str(train_set["user_ID"].unique().shape[0]))
    print("Train Movies " + str(train_set["movie_ID"].unique().shape[0]))
    print("Test Users " + str(test_set["user_ID"].unique().shape[0]))
    print("Test Movies " + str(test_set["movie_ID"].unique().shape[0]))
    
    print(train_set.shape[0])
    print(test_set.shape[0])
    
    test_set.to_csv(join(save_folder, "test_set.csv"))
    
    return

if __name__ == "__main__":
    select_subset(train_path = "dataset frac=0.33, ratio=4, min_samples=100/train_set.csv",
                  test_path = "dataset frac=0.33, ratio=4, min_samples=100/test_set.csv",
                  min_samples = 500, save_folder = "dataset frac=0.33, ratio=4, min_samples=250")
    
    
                        
    
    
