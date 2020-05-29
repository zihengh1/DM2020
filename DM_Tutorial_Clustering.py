import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

raw_data = pd.read_csv("./EmployeeAttrition.csv", index_col=0)

two_mode_data = np.array(raw_data[["TotalWorkingYears", "MonthlyIncome"]])

kmeans_model = KMeans(n_clusters=3, random_state=100).fit(two_mode_data)
print(kmeans_model.labels_)

gmm_model = GaussianMixture(n_components=3, random_state=0).fit(two_mode_data)
gmm_labels = gmm_model.predict(two_mode_data)
print(gmm_labels)