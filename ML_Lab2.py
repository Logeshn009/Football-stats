# uploading files

from google.colab import files
uploaded=files.upload()

import pandas as pd
import io
df=pd.read_csv(io.BytesIO(uploaded['isl_player_final.csv']))
print(df)

# replacing null values
df.isnull().any().any()
df = df.fillna(0)

import numpy as np
df.replace(to_replace=np.nan,value=0)

# Remove columns with all na values
all_na = df.columns[df.isna().all()]
df.drop(all_na, axis = 1, inplace = True)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

x = pd.read_csv("isl_player_final.csv", usecols = ['events.shots_on_target','events.goals'])
x
kmeans = KMeans(3)
kmeans.fit(x)
identified_clusters = kmeans.fit_predict(x)
identified_clusters
df['events.shots_on_target'] = df['events.shots_on_target'].astype(str)
df['events.goals'] = df['events.goals'].astype(str)
data_with_clusters = df.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['events.shots_on_target'],data_with_clusters['events.goals'],c=data_with_clusters['Clusters'],cmap='rainbow')

wcss=[]
for i in range(1,7):
  kmeans = KMeans(i)
  kmeans.fit(x)
  wcss_iter = kmeans.inertia_
  wcss.append(wcss_iter)


number_clusters = range(1,7)
plt.plot(number_clusters,wcss)
plt.title('The Elbow title')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
