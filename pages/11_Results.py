import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

file_name = Path('data/Metrics.csv')
df_Metrics=pd.read_csv(file_name, index_col=0)
cols = list(df_Metrics.keys())

st.header('Metrics for all models:')
st.table(df_Metrics)
#### GRAPH THE RESULTS ###


# Reshape the dataframe into long format using pd.melt()

subset_df = pd.melt(df_Metrics[cols].reset_index(), id_vars='Index', var_name='Model', value_name='Score')


sns.set_style('whitegrid')
ax=sns.catplot(data=subset_df, 
    x='Index', 
    y='Score', 
    hue='Model', 
    kind='bar', 
    palette='Blues', 
    aspect=2)

plt.xlabel('Clusters')
plt.ylabel('Scores')
plt.xticks(rotation = 45)

fig = ax.figure
st.pyplot(fig)