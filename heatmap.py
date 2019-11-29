# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:57:41 2019

@author: LENOVO
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

sns.palplot(sns.diverging_palette(11, 150, n = 10, center = 'dark'))


df = pd.read_csv(r"C:\Users\ahls_st\Documents\MasterThesis\Results_CSVs\Single_augs.csv")
mask = df.loc[df["Network"]=="Mask RCNN"]
mask = mask.loc[:, "Augmentation":"Recall (%)"]
mask = mask.set_index("Augmentation", drop = True)
sns.set(font_scale=1.9)

cm = np.array(['Greens', "Greens", "Greens"])

f, axs = plt.subplots(1, mask.columns.size, gridspec_kw={'wspace': 0}, figsize=(13, 10))
f.suptitle(r'$\bf{Individual}$'+' '+r'$\bf{Augmentations}$'+' '+r'$\bf{(Mask}$'+' '+r'$\bf{R}$'+r'$\bf{-}$'+r'$\bf{CNN)}$', size = 25, x=0.6) 
for i, (s, a, c) in enumerate(zip(mask.columns, axs, cm)):
    
    if i ==2:
         sns.heatmap(np.array([mask[s].values*100]).T, yticklabels=mask.index, 
                xticklabels=[s], annot=True, fmt='.1f', ax=a, cmap=c, cbar=True,  cbar_kws={'ticks': [np.min(mask[s].values*100), np.max(mask[s].values*100)]})
         colorbar = a.collections[0].colorbar
         colorbar.set_ticklabels(['Low', 'High'])
    sns.heatmap(np.array([mask[s].values*100]).T, yticklabels=mask.index, 
                xticklabels=[s], annot=True, fmt='.1f', ax=a, cmap=c, cbar=False)
    if i>0:
        a.yaxis.set_ticks([])
        
plt.tight_layout(rect=(0, 0, 0.96, 0.95))
plt.savefig('heatmap_mask.png', dpi=500)




# Repeat for DeepLab
deep = df.loc[df["Network"]=="DeepLab v3+"]
deep = deep.loc[:, "Augmentation":"Recall (%)"]
deep = deep.set_index("Augmentation", drop = True)


f, axs = plt.subplots(1, deep.columns.size, gridspec_kw={'wspace': 0}, figsize=(13, 10))
f.suptitle(r'$\bf{Individual}$'+' '+r'$\bf{Augmentations}$'+' '+r'$\bf{(DeepLab}$'+' '+r'$\bf{v3}$'+r'$\bf{+)}$', size = 25, x=0.6) 
for i, (s, a, c) in enumerate(zip(deep.columns, axs, cm)):
    if i==2:
        sns.heatmap(np.array([deep[s].values*100]).T, yticklabels=deep.index,
                xticklabels=[s], annot=True, fmt='.1f', ax=a, cmap=c, cbar=True, cbar_kws={'ticks': [np.min(deep[s].values*100), np.max(deep[s].values*100)]})
        colorbar = a.collections[0].colorbar
        colorbar.set_ticklabels(['Low', 'High'])
        
    else:
        sns.heatmap(np.array([deep[s].values*100]).T, yticklabels=deep.index,
                xticklabels=[s], annot=True, fmt='.1f', ax=a, cmap=c, cbar=False)
    if i>0:
        a.yaxis.set_ticks([])


plt.tight_layout(rect=(0, 0, 0.96, 0.95))
plt.savefig('heatmap_deep.png', dpi=500)

############### Band Comps


df = pd.read_csv(r"C:\Users\ahls_st\Documents\MasterThesis\Results_CSVs\Band_comps.csv")

mask = df.loc[df["Network"]=="Mask RCNN"]
mask = mask.loc[:, "Band comp":"Recall (30%)"]
mask = mask.set_index("Band comp", drop = True)
sns.set(font_scale=1.4)

cm = np.array(['Blues_r', "Reds_r", "Greens_r", "Reds_r", "Greens_r", "Reds_r", "Greens_r"])
f, axs = plt.subplots(1, mask.columns.size, gridspec_kw={'wspace': 0}, figsize=(17, 8))
f.suptitle(r'$\bf{Band}$'+' '+r'$\bf{Combinations}$'+' '+r'$\bf{(Mask}$'+' '+r'$\bf{R}$'+r'$\bf{-}$'+r'$\bf{CNN)}$', size = 25,  x=0.6) 
for i, (s, a, c) in enumerate(zip(mask.columns, axs, cm)):
    sns.heatmap(np.array([mask[s].values]).T, yticklabels=mask.index, 
                xticklabels=[s], annot=True, fmt='.3f', ax=a, cmap=c, cbar=False)
    if i>0:
        a.yaxis.set_ticks([])
plt.tight_layout(rect=(0, 0, 0.96, 0.95))
plt.savefig('heatmap_bands_mask.png', dpi=500)




# Repeat for DeepLab
deep = df.loc[df["Network"]=="DeepLab v3+"]
deep = deep.loc[:, "Band comp":"Recall (30%)"]
deep = deep.set_index("Band comp", drop = True)

cm = np.array(['Blues_r', "Reds_r", "Greens_r", "Reds_r", "Greens_r", "Reds_r", "Greens_r"])
f, axs = plt.subplots(1, deep.columns.size, gridspec_kw={'wspace': 0}, figsize=(17, 8))
f.suptitle(r'$\bf{Band}$'+' '+r'$\bf{Combinations}$'+' '+r'$\bf{(DeepLab}$'+' '+r'$\bf{v3+)}$', size = 25) 
for i, (s, a, c) in enumerate(zip(deep.columns, axs, cm)):
    sns.heatmap(np.array([deep[s].values]).T, yticklabels=deep.index,
                xticklabels=[s], annot=True, fmt='.3f', ax=a, cmap=c, cbar=False)
    if i>0:
        a.yaxis.set_ticks([])


plt.savefig('heatmap_bands_deep.png', dpi=500)