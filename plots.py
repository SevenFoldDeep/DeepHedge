# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:35:31 2019

@author: ahls_st
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

values = pd.read_csv(r'C:\Users\ahls_st\Documents\MasterThesis\Results_CSVs\Single_augs.csv')
values = values.rename(columns={"Per-pixel accuracy": "PPA"})
mask = values[values['Network'] == 'Mask RCNN']
deep = values[values['Network'] == 'DeepLab v3+']
col_select = mask.loc[:21,'PPA': 'Recall30'] #change between mask.loc and deep.loc to get results for diff networks
values1 = []
for row in col_select.itertuples():
    values1.append(row.PPA*100)
    values1.append(row.Precision70*100)
    values1.append(row.Precision50*100)
    values1.append(row.Precision30*100)
    values1.append(row.Recall70*100)
    values1.append(row.Recall50*100)
    values1.append(row.Recall30*100)
    values1.append(0)
    values1.append(0)

l = len(values1)

#need to add label to my data
#plt.plot

colorlist=np.array(['#800000', '#99ff33', '#33cc33', '#006600','#66ccff', '#0099ff', '#003399', 'w', 'w']*11)
labs = np.array(['Pixel Accuracy', 'Precision @70%', 'Precision @50%', 'Precision @30%', 'Recall @70%', 'Recall @50%', 'Recall @30%', '', '']*11)

theta=np.arange(0,2*np.pi,2*np.pi/l)
width = (2*np.pi)/l *0.9
bottom = 50

fig = plt.figure(figsize=(9,9))

#Choose appropriate title
fig.suptitle(r'$\bf{Mask}$'+' '+r'$\bf{R}$'+r'$\bf{-}$'+r'$\bf{CNN}$'+' '+r'$\bf{Augmentations}$', size = 25) 
#fig.suptitle(r'$\bf{DeepLab}$'+' '+r'$\bf{v3+}$'+ ' '+ r'$\bf{Augmentations}$', size = 25) 


ax = fig.add_axes([0.1, 0.1, 0.75, 0.75], polar=True)
bars = ax.bar(theta, values1, width=width, bottom=bottom, color=colorlist)
for i, bar in enumerate(bars):
    if i == 7:
        break
    bar.set_label(labs[i])
#bars.set_color('r')

plt.axis('off')
plt.legend()
rotations = np.rad2deg(theta)

i=-1
for x, bar, rotation, label in zip(theta, bars, rotations, values1):
    i+=1
    if int(label) == 0:
        continue
    if i in range(0, 9):
        lab = ax.text(x-.025,bottom+bar.get_height()+15 , round(label, 1), 
            ha='right', va='center', size=12)
    elif i in range(9, 18):
        lab = ax.text(x-.025,bottom+bar.get_height()+15 , round(label, 1), 
            ha='right', va='center', size=12, rotation=30, rotation_mode="anchor")
    elif i in range(18, 27):
        lab = ax.text(x-.09,bottom+bar.get_height()+12 , round(label, 1), 
            ha='right', va='center', size=12, rotation=30, rotation_mode="anchor")
    elif i in range(27, 29):
        lab = ax.text(x,bottom+bar.get_height()+1 , round(label, 1), 
            ha='right', va='center', size=12, rotation=280, rotation_mode="anchor")
    elif i in range(29, 35):
        lab = ax.text(x,bottom+bar.get_height()+1 , round(label, 1), 
            ha='right', va='center', size=12, rotation=293, rotation_mode="anchor")
    elif i in range(35, 37):
        lab = ax.text(x,bottom+bar.get_height()+1 , round(label, 1), 
            ha='right', va='center', size=12, rotation=315, rotation_mode="anchor")
    elif i in range(37, 42):
        lab = ax.text(x,bottom+bar.get_height()+1 , round(label, 1), 
            ha='right', va='center', size=12, rotation=322, rotation_mode="anchor")
    elif i in range(42, 46):
        lab = ax.text(x,bottom+bar.get_height() , round(label, 1), 
             ha='right', va='center', size=12, rotation=345, rotation_mode="anchor")
    elif i in range(46, 49):
        lab = ax.text(x,bottom+bar.get_height() , round(label, 1), 
             ha='right', va='center', size=12, rotation=355, rotation_mode="anchor")
    elif i in range(49, 52):
        lab = ax.text(x,bottom+bar.get_height() , round(label, 1), 
             ha='right', va='center', size=12)
    elif i in range(52, 61):
        lab = ax.text(x,bottom+bar.get_height() , round(label, 1), 
             ha='right', va='center', size=12, rotation=20, rotation_mode="anchor")
    elif i in range(63, 66):
        lab = ax.text(x+0.002,bottom+bar.get_height() , round(label, 1), 
             ha='right', va='center', size=12, rotation=48, rotation_mode="anchor")
    elif i in range(66, 70):
        lab = ax.text(x+0.048,bottom+bar.get_height() , round(label, 1), 
             ha='right', va='bottom', size=12, rotation=53, rotation_mode="anchor")
    elif i in range(71, 79):
        lab = ax.text(x+.042,bottom+bar.get_height() , round(label, 1), 
             ha='right', va='bottom', size=12, rotation=90, rotation_mode="anchor")
    elif i in range(79, 84):
        lab = ax.text(x-.05,bottom+bar.get_height()+13.5, round(label, 1), 
             ha='right', va='bottom', size=12, rotation=297, rotation_mode="anchor")
    elif i in range(84, 86):
        lab = ax.text(x-.047,bottom+bar.get_height()+13.5, round(label, 1), 
             ha='right', va='bottom', size=12, rotation=305, rotation_mode="anchor")
    elif i in range(86, 89):
        lab = ax.text(x-.045,bottom+bar.get_height()+13.7, round(label, 1), 
             ha='right', va='bottom', size=12, rotation=310, rotation_mode="anchor")
    elif i in range(89, 97):
        lab = ax.text(x-.045,bottom+bar.get_height()+13.7, round(label, 1), 
             ha='right', va='bottom', size=12, rotation=335, rotation_mode="anchor")
    else:
        lab = ax.text(x,bottom+bar.get_height() , round(label, 1), 
             ha='center', va='center', size=12)
    if i == 3:
        lab = ax.text(x-0.05,bottom+bar.get_height()+34 , r'$\bf{None}$', 
             ha='center', va='center', size=16)
    if i == 12:
        lab = ax.text(x,bottom+bar.get_height()+40 , r'$\bf{Add}$', 
             ha='center', va='center', size=16)
    if i == 21:
        lab = ax.text(x-0.3,bottom+bar.get_height()+45 , r'$\bf{Add}$'+' '+r'$\bf{elementwise}$', 
             ha='center', va='center', size=16, rotation = 0, rotation_mode="anchor")
    if i == 30:
        lab = ax.text(x,bottom+bar.get_height()+37 , r'$\bf{Blur}$', 
             ha='center', va='center', size=16)
    if i == 39:
        lab = ax.text(x,bottom+bar.get_height()+40 , r'$\bf{Add}$'+' '+r'$\bfper}$'+' '+r'$\bf{channel}$', 
             ha='center', va='center', size=16, rotation=0, rotation_mode="anchor")
    if i == 48:
        lab = ax.text(x,bottom+bar.get_height()+40 , r'$\bf{Contrast}$', 
             ha='center', va='center', size=16, rotation=0, rotation_mode="anchor")
    if i == 57:
        lab = ax.text(x-.2,bottom+bar.get_height()+64 , r'$\bf{Contrast}$'+' '+r'$\bfper}$'+' '+r'$\bf{channel}$', 
             ha='center', va='center', size=16, rotation=0, rotation_mode="anchor")
    if i == 66:
        lab = ax.text(x,bottom+bar.get_height()+30 , r'$\bf{Flip}$', 
             ha='center', va='center', size=16, rotation=0, rotation_mode="anchor")
    if i == 75:
        lab = ax.text(x,bottom+bar.get_height()+38 , r'$\bf{Rotate}$'+' '+r'$\bf{45°}$', 
             ha='center', va='center', size=16, rotation=0, rotation_mode="anchor")
    if i == 84:
        lab = ax.text(x,bottom+bar.get_height()+40 , r'$\bf{Rotate}$'+' '+r'$\bf{90°}$', 
             ha='center', va='center', size=16, rotation=0, rotation_mode="anchor")
    if i == 93:
        lab = ax.text(x,bottom+bar.get_height()+40 , r'$\bf{Scale}$', 
             ha='center', va='center', size=16, rotation=0, rotation_mode="anchor")
    
plt.legend(loc='center')
plt.show()