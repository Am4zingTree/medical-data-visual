import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def draw_cat_plot():
    
    df = pd.read_csv("medical_examination.csv")

    # BMI = weight(kg) / (height(m))^2
    bmi = df['weight'] / ((df['height'] / 100) ** 2)
    df['overweight'] = (bmi > 25).astype(int)

    df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
    df['gluc'] = (df['gluc'] > 1).astype(int)

    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    catplot = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar',
        height=5,
        aspect=1
    )

    catplot.set_axis_labels("variable", "total")
    catplot._legend.set_title("value")

    fig = catplot.fig

    return fig


def draw_heat_map():

    df = pd.read_csv("medical_examination.csv")

    # Add 'overweight' and normalize 'cholesterol' and 'gluc' as in the cat plot
    bmi = df['weight'] / ((df['height'] / 100) ** 2)
    df['overweight'] = (bmi > 25).astype(int)
    df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
    df['gluc'] = (df['gluc'] > 1).astype(int)


    # Keep rows where ap_lo <= ap_hi
    df_heat = df[df['ap_lo'] <= df['ap_hi']].copy()

    # Keep rows where height and weight are between 2.5th and 97.5th percentiles
    height_low = df_heat['height'].quantile(0.025)
    height_high = df_heat['height'].quantile(0.975)
    weight_low = df_heat['weight'].quantile(0.025)
    weight_high = df_heat['weight'].quantile(0.975)

    df_heat = df_heat[
        (df_heat['height'] >= height_low) &
        (df_heat['height'] <= height_high) &
        (df_heat['weight'] >= weight_low) &
        (df_heat['weight'] <= weight_high)
    ].copy()

    corr = df_heat.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )

    return fig
