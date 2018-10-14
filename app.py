import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

kbest = SelectKBest(mutual_info_regression, k=10 )
print("Starting file load.")
train_df = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
print("Files loaded!")
train_df, test_df = train_test_split(
    train_df
)
print("Split")
# Drop ids
print("Dropping columns: ", train_df.columns[0:3])
train_df = train_df.drop(train_df.columns[0:3], axis=1)
# Create a correlation df
training_correlation = train_df.corr()[
    'winPlacePerc'
].abs()
unacceptable_correlation = 0.25
print("Dropping "+str(training_correlation[training_correlation < unacceptable_correlation])+" due to correlation being beneath: "+str(unacceptable_correlation))
train_df = train_df.drop(
    training_correlation[training_correlation < unacceptable_correlation].index,
    axis = 1
)
print("Starting Loop")
# results = np.ndarray()
df_sample = train_df.sample(1000)
z = mutual_info_regression(df_sample[df_sample.columns[:-1]].as_matrix(), df_sample[df_sample.columns[-1]].as_matrix())
for x in range(0, 1000):
    print("Iteration: ", x)
    df_sample = train_df.sample(5000) 
    z = np.vstack([z, mutual_info_regression(df_sample[df_sample.columns[:-1]].as_matrix(), df_sample[df_sample.columns[-1]].as_matrix())])

for x in range(0,14):
    print(df_sample.columns[x],z[:,:].mean(axis=0)[x])
