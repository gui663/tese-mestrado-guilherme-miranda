import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./CSV/07092022_121214.csv')

data = df.drop(labels=['Label', 'Type', 'Name'], axis='columns')
print(data.head())
data = MinMaxScaler().fit_transform(data)
#data = RobustScaler().fit_transform(data)
#data = StandardScaler().fit_transform(data)
labels = df.loc[:, "Label"]

pca = PCA(n_components=3)
pca_result = pca.fit_transform(data)

pca_one= pca_result[:,0]
pca_two= pca_result[:,1] 
pca_three= pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
plt.figure(1, figsize=(8,10))
#plt.subplot(311)
#sns.scatterplot(
#    x=pca_one, y=pca_two,
#    hue = labels,
#    palette=sns.color_palette("hls", 2),
#    data=data,
#    legend="full",
#    alpha=0.3
#)

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data)
tsne_2d_one= tsne_results[:,0]
tsne_2d_two= tsne_results[:,1]

plt.plot()
sns.scatterplot(
    x=tsne_2d_one, y=tsne_2d_two,
    hue=labels,
    palette=('navy', 'forestgreen'),
    data=data,
    legend=0,
    alpha=0.3
)

ax = plt.gca()

ax.axes.xaxis.set_ticklabels([])

ax.axes.yaxis.set_ticklabels([])
#plt.subplot(313)
#plt.boxplot(data)

plt.show()