import pandas as pd
from datetime import datetime as dt
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np
from sklearn.neighbors import NearestNeighbors
import streamlit as st

df = pd.read_csv("data/VCPe SevOne Data 10_7 10_8 7am with Deltas and disk.csv")


"""

# TODO: convert 'time' column to date/time
# pd.to_datetime(data['time'], infer_datetime_format=True)
# index = pd.to_datetime(df['TIME'], unit='s', format="%Y-%m-%d %H")
index = pd.to_datetime(df['TIME'], unit='s').dt.strftime("%Y-%m-%d %H:00")

data = df[['DEVICENAME', 'SSCPUIDLE']].set_index(index)

device = data[data.DEVICENAME == "ALPRGAGQNCE-H-PE1C7CN-001"]

sns.distplot(data.SSCPUIDLE)
sns.histplot(data=data.SSCPUIDLE)




# from tslearn.clustering import TimeSeriesKMeans
# km = TimeSeriesKMeans(n_clusters=3, metric="dtw")

# labels = km.fit_predict(data)


km_bis = TimeSeriesKMeans(n_clusters=2, metric="softdtw")
labels_bis = km_bis.fit_predict(X







from tslearn.utils import to_time_series_dataset
X = to_time_series_dataset([[1, 2, 3, 4], [1, 2, 3], [2, 5, 6, 7, 8, 9]])
y = [0, 0, 1]

from tslearn.clustering import KernelKMeans
gak_km = KernelKMeans(n_clusters=2, kernel="gak")
labels_gak = gak_km.fit_predict(X)

from tslearn.clustering import TimeSeriesKMeans
km = TimeSeriesKMeans(n_clusters=2, metric="dtw")
labels = km.fit_predict(X)
km_bis = TimeSeriesKMeans(n_clusters=2, metric="softdtw")
labels_bis = km_bis.fit_predict(X


# sns.scatterplot(data=data.SSCPUIDLE, x="DEVICENAME", y="SSCPUIDLE")

# km = KMeans()
# x = cpu_data.value.fillna(0).values.reshape(-1, 1)
# clusters = km.fit_predict(x)
# cpu_data['cluster'] = clusters.T
# sns.scatterplot(data=cpu_data, x="cluster", y="value", hue="cluster")






# kmeans = KMeans(n_clusters=2, random_state=0).fit(cpu_data.value)
# gb_data = cpu_data.groupby('deviceId')
# for device, df in gb_data:
    # print(device)
    # print(df)

# x = np.random.normal(0, 1, 1000)
# ax = sns.distplot(x)

"""