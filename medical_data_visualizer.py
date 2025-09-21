import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def draw_cat_plot():
    # 1) Import the data
    df = pd.read_csv("medical_examination.csv")

    # 2) Add 'overweight' column
    # BMI = weight(kg) / (height(m))^2
    bmi = df['weight'] / ((df['height'] / 100) ** 2)
    df['overweight'] = (bmi > 25).astype(int)

    # 3) Normalize 'cholesterol' and 'gluc' to 0 (good) and 1 (bad)
    df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
    df['gluc'] = (df['gluc'] > 1).astype(int)

    # 4) Draw the categorical plot
    # 5) Create DataFrame for cat plot using pd.melt()
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6) Group and reformat the data to show counts of each feature split by cardio
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7) Draw the catplot using seaborn
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

    # 8) Get the figure for the output
    fig = catplot.fig

    # 9) Do not modify the next two lines
    # (The tests expect the function to return the figure)
    return fig


def draw_heat_map():
    # 10) Import the data
    df = pd.read_csv("medical_examination.csv")

    # Add 'overweight' and normalize 'cholesterol' and 'gluc' as in the cat plot
    bmi = df['weight'] / ((df['height'] / 100) ** 2)
    df['overweight'] = (bmi > 25).astype(int)
    df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
    df['gluc'] = (df['gluc'] > 1).astype(int)

    # 11) Clean the data
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

    # 12) Calculate the correlation matrix
    corr = df_heat.corr()

    # 13) Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14) Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15) Draw the heatmap with sns.heatmap()
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

    # 16) Do not modify the next two lines
    return fig
