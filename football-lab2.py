# uploaded dataset

from google.colab import files

uploaded = files.upload()

#importing needed packages

import pandas as pd
import numpy as ny
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import pandas as pd 

pd.set_option('display.max_columns',100)
# reading the csv file

data = pd.read_csv('isl_player_final.csv')
#1 .shape attribute (dimensions) of the dataframe - which gives the overall structure of the data

data.shape #output (563, 93) - (rows, cols)

#2 data types of the various columns
data.dtypes

#3 Display a few rows

data.head()
data.tail()
data.sample(5)
