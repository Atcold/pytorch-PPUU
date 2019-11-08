# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import PCA
import Widgets
import DataReader
import VisualizationLibrary
import importlib
import Tabs
from types import ModuleType

importlib.reload(DataReader)
importlib.reload(PCA)
importlib.reload(Widgets)
importlib.reload(Tabs)
importlib.reload(VisualizationLibrary)

v = VisualizationLibrary.Visualization()
v.display()

# %%
import sklearn
import sklearn.cluster
from matplotlib import pyplot as plt

features = PCA.PCA.get_pca_data()
scores = []

for c in range(2, 15):
    score = sklearn.cluster.KMeans(n_clusters=c).fit(features).score(features)
    scores.append(-score)

plt.plot(scores, 'o-')




# %%
