# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from jupyterthemes import jtplot

# %%
jtplot.style('oceans16')

# %%
act_grads = torch.load('../actions_grads_orig.pkl')
data = [np.array(k) for k in act_grads]
data = np.concatenate(data, axis=0)
xedges = [np.quantile(data[:,0], p) for p in np.linspace(0,1,21)]

df = pd.DataFrame(data)
df.columns = ['speed', 'grad_proximity', 'grad_lane']
df.speed = (df.speed//5 * 5).astype(int)

# %% [markdown]
# ## Original Implementation

# %%
df_new = df.copy()

# %%
df_new.boxplot(column='grad_proximity', by='speed', figsize=(14,8),)

# %%
df_new.boxplot(column='grad_lane', by='speed', figsize=(14,8))

# %%
df_new[['grad_proximity', 'grad_lane']] = np.log10(df_new[['grad_proximity', 'grad_lane']])

# %%
df_new.boxplot(column='grad_proximity', by='speed', figsize=(14,8))

# %%
df_new.boxplot(column='grad_lane', by='speed', figsize=(14,8))

# %% [markdown]
# ## Constant Speed

# %%
act_grads = torch.load('../actions_grads.pkl')
data = [np.array(k) for k in act_grads]
data = np.concatenate(data, axis=0)
xedges = [np.quantile(data[:,0], p) for p in np.linspace(0,1,21)]

df = pd.DataFrame(data)
df.columns = ['speed', 'grad_proximity', 'grad_lane']
df.speed = (df.speed//5 * 5).astype(int)

# %%
df_new = df.copy()

# %%
df_new.boxplot(column='grad_proximity', by='speed', figsize=(14,8),)

# %%
df_new.boxplot(column='grad_lane', by='speed', figsize=(14,8))

# %%
df_new[['grad_proximity', 'grad_lane']] = np.log10(df_new[['grad_proximity', 'grad_lane']])

# %%
df_new.boxplot(column='grad_proximity', by='speed', figsize=(14,8))

# %%
df_new.boxplot(column='grad_lane', by='speed', figsize=(14,8))

# %%
