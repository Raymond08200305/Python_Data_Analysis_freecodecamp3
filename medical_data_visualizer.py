import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')
print(df.shape)
print(df.columns.tolist())
print(df.head())

# 2
bmi = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = (bmi > 25).astype(int)

# 3
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5
    features = ['active', 'alco', 'cholesterol',
                'gluc', 'overweight', 'smoke']
    df_cat = pd.melt(df,id_vars=['cardio'],value_vars=features,
                     var_name='variable',value_name='value')

    # 6
    df_cat = (df_cat.groupby(['cardio','variable','value']).size().reset_index(name='total'))
    

    # 7
    g = sns.catplot(data=df_cat,x='variable', y='total', 
                    hue='value',col='cardio', kind='bar',
                    order=features, hue_order=[0, 1],errorbar=None)

    # 8
    fig = g.figure


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    height_low = df['height'].quantile(0.025)
    height_high = df['height'].quantile(0.975)
    weight_low = df['weight'].quantile(0.025)
    weight_high = df['weight'].quantile(0.975)
    valid = ((df['ap_lo'] <= df['ap_hi']) & (df['height'] >= height_low) & 
             (df['height'] <= height_high) & (df['weight'] >= weight_low) & 
             (df['weight'] <= weight_high))
    
    df_heat = df.loc[valid].copy()


    # 12
    corr = df_heat.corr(numeric_only=True)

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15
    sns.heatmap(corr,mask=mask,annot=True,fmt='.1f',cmap='coolwarm',
                center=0,square=True,linewidths=0.4,ax=ax,
                cbar_kws={'shrink': 0.75})

    # 16
    fig.savefig('heatmap.png')
    return fig
