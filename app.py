import streamlit as st
import joblib
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

data, ts_data, = joblib.load("data/data.job")

location = st.sidebar.selectbox('Location:', ts_data['Location'].unique())

clusters = st.sidebar.slider('Clusters:', 2, 6)

ts = to_time_series_dataset(ts_data[ts_data.Location == location].TimeSeries.values)

st.subheader(f"Location: {location}, Devices: {len(ts)}, Clusters: {clusters}")
st.text("")

km = TimeSeriesKMeans(n_clusters=clusters, metric="dtw", n_jobs=7)
labels = km.fit_predict(ts)

df = ts_data[ts_data.Location == location].copy()
df['Cluster'] = labels.T

for cluster in np.sort(np.unique(labels)):
    cdf = df[df.Cluster == cluster]
    for k, s in cdf.TimeSeries.items():
        s = pd.DataFrame(s, columns=['CPU Idle'])
        s['Hour'] = s.index
        sns.lineplot(data=s, x="Hour", y="CPU Idle", alpha=0.1)
    sns.lineplot(data=km.cluster_centers_[cluster], legend=False)
    plt.title(f"Cluster: {cluster}", size=12)
    st.pyplot()
    # st.dataframe(cdf[['Device', 'Day']].reset_index(drop=True), width=500)

st.table(df[['Device', 'Day', 'Cluster']].sort_values(by=['Cluster']).reset_index(drop=True))


