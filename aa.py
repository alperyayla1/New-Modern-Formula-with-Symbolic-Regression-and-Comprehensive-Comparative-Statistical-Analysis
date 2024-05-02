import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functionssmoking import *
dataset = pd.read_excel("C:/Users/alper/OneDrive/Masaüstü/Kitap1.xlsx")

dataset.drop(dataset.columns[4], axis=1, inplace=True)
dataset.dropna(how='any', inplace=True, axis=0)

print("Number of samples: ", len(dataset))

df_sorted_heart_rate = dataset.sort_values(by='heart_rate')
dataset = drop_outliers(df_sorted_heart_rate, 'heart_rate')

df_sorted_chol = dataset.sort_values(by='chol')
print(df_sorted_chol)
dataset = drop_outliers(df_sorted_chol, 'col')
print(dataset)



#plt.hist(dataset_filtered['age'], dataset_filtered['chol'])