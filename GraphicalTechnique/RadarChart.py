import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# read dataset from csv and perform preprocessing
data = pd.read_csv('ScoreData.csv')

categories = ['SE','AI','OOP','MSA',
            'OS','DHMT','XLAV','VLDC2']
# plot unfilled scatter plot
fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=pd.Series(data.loc[0].values),
      theta=categories,
      fill='toself',
      name='Student A'
))
fig.add_trace(go.Scatterpolar(
    r=pd.Series(data.loc[1].values),
      theta=categories,
      fill='toself',
      name='Student B'
))
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
        ),
    ),
    template='plotly_dark',
    showlegend=True,

)
fig.show()
