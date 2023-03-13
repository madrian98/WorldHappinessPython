#!/usr/bin/env python
# coding: utf-8

# # 1.Environment

# ## 1.1 Importing  libraries for data analysis/mining

# In[2]:


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import folium
import pycountry
import pycountry_convert as pc
import geopandas as gpd
import ipywidgets as widgets
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.graph_objs as go
from chart_studio import plotly
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from geopy.geocoders import Nominatim
from IPython.display import display, HTML
from folium.plugins import MarkerCluster
pyo.init_notebook_mode()
init_notebook_mode(connected=True)
display(HTML("<style>.container { width:100% !important; }</style>"))

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 50)
warnings.filterwarnings("ignore")


# ### Importing libraries/modules for machine learning

# In[3]:


import sklearn
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import ensemble, tree, linear_model,cluster, mixture,datasets
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet,Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error,r2_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.cluster import silhouette_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


# # 2 Data

# ## 2.1 Data import

# ### 2021 data

# In[4]:


df_2021= pd.read_csv('data/2021.csv')


# ### Data info

# In[5]:


df_2021.info();


# ### Data description

# In[6]:


df_2021.describe()


# ### Searching for an empty values

# In[7]:


df_2021.isna().sum()


# ### Previewing data from 2021

# In[8]:


df_2021.head(10)


# ## 2.2 Data info

# ### 2021 data

# In[9]:


df_2021.info();


# Variables:
# - Country name.
# - Regional indicator.
# - Ladder score(Happiness score).
# - Standard error of ladder score(happiness score) : standard error of happiness score.
# - Upperwhisker : upperwhisker of happiness score.
# - Lowerwhisker : lowerwhistker of happiness score.
# - Logged GDP per capita: in described country.
# - Social support : in described country.
# - Healthy life expectancy : in described country.
# - Freedom to make life choices : in described country.
# - Generosity : in described country.
# - Perceptions of corruption : in described country.
# - Ladder score in Dystopia(Happiness score) : Dystopia - hyphothetic country with score lower than lowest in dataset.
# - Explained by: Log GDP per capita.
# - Explained by: Social support. 
# - Explained by: Healthy life expectancy.
# - Explained by: Freedom to make life choices.
# - Explained by: Generosity.
# - Explained by: Perceptions of corruption.
# - Dystopia + residual : Residual - other factors which couldn't be grouped as one for described country.

# # 3.Data mining

# In[10]:


df_2021.rename({"Ladder score": "Happiness score"},axis=1, inplace=True)
df_2021.rename({"Ladder score in Dystopia": "Happiness score in Dystopia"},axis=1, inplace=True)


# ## Number of countries for each region

# In[11]:


df_regions = df_2021.groupby('Regional indicator').agg({'Country name':'count'}).reset_index()
fig = px.pie(values = df_regions['Country name'],
             names= df_regions['Regional indicator'],
             title="% of countries in each region",
             color_discrete_sequence=px.colors.qualitative.Dark2,
             )
fig.update_layout(title_font_size=22,margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000,legend_title_text='Region')
fig.show()


# ## Variabless in each region

# In[12]:


fig = px.box(df_2021,
             x="Happiness score",
             y="Regional indicator",
             color="Regional indicator",
             color_discrete_sequence=px.colors.qualitative.Pastel_r,
             template="plotly_white")
fig.update_traces(boxmean=True,
                  whiskerwidth=0.8,
                  marker_size=2,
            line_width=2.5
                  )
fig.update_layout(title='Happiness score in each region',title_font_size=22,margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000,legend_title_text='Region',xaxis_title_text='Happiness score',yaxis_title_text='Region')
fig.show()


# In[13]:


fig = px.box(df_2021,
             x="Logged GDP per capita",
             y="Regional indicator",
             color="Regional indicator",
             color_discrete_sequence=px.colors.qualitative.Pastel_r,
             template="plotly_white")
fig.update_traces(boxmean=True,
                  whiskerwidth=0.8,
                  marker_size=2,
            line_width=2.5
                  )
fig.update_layout(title='Logged GDP per capita in each region',title_font_size=22,margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000,legend_title_text='Region',xaxis_title_text='Logged GDP per capita',yaxis_title_text='Region')
fig.show()


# In[14]:


fig = px.box(df_2021,
             x="Social support",
             y="Regional indicator",
             color="Regional indicator",
             color_discrete_sequence=px.colors.qualitative.Pastel_r,
             template="plotly_white")
fig.update_traces(boxmean=True,
                  whiskerwidth=0.8,
                  marker_size=2,
            line_width=2.5
                  )
fig.update_layout(title='Social support in each region',title_font_size=22,margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000,legend_title_text='Region',xaxis_title_text='Social support',yaxis_title_text='Region')
fig.show()


# In[15]:


fig = px.box(df_2021,
             x="Healthy life expectancy",
             y="Regional indicator",
             color="Regional indicator",
             color_discrete_sequence=px.colors.qualitative.Pastel_r,
             template="plotly_white")
fig.update_traces(boxmean=True,
                  whiskerwidth=0.8,
                  marker_size=2,
            line_width=2.5
                  )
fig.update_layout(title='Healthy life expectancy in each region',title_font_size=22,margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000,legend_title_text='Region',xaxis_title_text='Healthy life expectancy',yaxis_title_text='Region')
fig.show()


# In[16]:


fig = px.box(df_2021,
             x="Freedom to make life choices",
             y="Regional indicator",
             color="Regional indicator",
             color_discrete_sequence=px.colors.qualitative.Pastel_r,
             template="plotly_white")
fig.update_traces(boxmean=True,
                  whiskerwidth=0.8,
                  marker_size=2,
            line_width=2.5
                  )
fig.update_layout(title='Freedom to make life choices in each region',title_font_size=22,margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000,legend_title_text='Region',xaxis_title_text='Freedom to make life choices',yaxis_title_text='Region')
fig.show()


# In[17]:


fig = px.box(df_2021,
             x="Generosity",
             y="Regional indicator",
             color="Regional indicator",
             color_discrete_sequence=px.colors.qualitative.Pastel_r,
             template="plotly_white")
fig.update_traces(boxmean=True,
                  whiskerwidth=0.8,
                  marker_size=2,
            line_width=2.5
                  )
fig.update_layout(title='Generosity in each region',title_font_size=22,margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000,legend_title_text='Region',xaxis_title_text='Generosity',yaxis_title_text='Region')
fig.show()


# In[18]:


fig = px.box(df_2021,
             x="Perceptions of corruption",
             y="Regional indicator",
             color="Regional indicator",
             color_discrete_sequence=px.colors.qualitative.Pastel_r,
             template="plotly_white")
fig.update_traces(boxmean=True,
                  whiskerwidth=0.8,
                  marker_size=2,
            line_width=2.5
                  )
fig.update_layout(title='Perceptions of corruption in each region',title_font_size=22,margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000,legend_title_text='Region',xaxis_title_text='Perceptions of corruption',yaxis_title_text='Region')
fig.show()


# In[19]:


fig = px.box(df_2021,
             x="Dystopia + residual",
             y="Regional indicator",
             color="Regional indicator",
             color_discrete_sequence=px.colors.qualitative.Pastel_r,
             template="plotly_white")
fig.update_traces(boxmean=True,
                  whiskerwidth=0.8,
                  marker_size=2,
            line_width=2.5
                  )
fig.update_layout(title='Dystopia + residual in each region',title_font_size=22,margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000,legend_title_text='Region',xaxis_title_text='Dystopia + residual',yaxis_title_text='Region')
fig.show()


# ### Averages Values for each region

# In[22]:


df_reg=df_2021.copy()
def feature_analysis(df_reg, feature):
    template='%{text:0.2f}'
    tickformat = None
    grouped_df = df_reg.groupby(["Regional indicator"]).agg({feature : np.mean}).reset_index()
    if grouped_df[feature].min() < 1:
        template='%{text:0.2%}'
        tickformat = ".2%"
        
    fig = px.bar(grouped_df,
                 x="Regional indicator",
                 y=feature,
                 color="Regional indicator",
                 text=feature,
                 color_discrete_sequence=px.colors.qualitative.D3,
                 template="plotly_dark"
                )

        
    fig.update_traces(texttemplate=template, 
                      textposition='outside', 
                      marker_line_color='rgb(255,255,255)', 
                      marker_line_width=2.5, 
                      opacity=0.7)

    fig.update_layout(showlegend=False,
                      title="{} in regions".format(feature),
                      yaxis=dict(tickformat=tickformat),xaxis_title_text='Region')

    fig.show()
    
    return None

feature_names = ["Logged GDP per capita",
                 "Social support",
                 "Healthy life expectancy",
                 "Freedom to make life choices",
                 "Generosity",
                 "Perceptions of corruption",
                "Dystopia + residual"]
for feature in feature_names:
    feature_analysis(df_reg, feature)


# ### Adding continents

# In[23]:


dfContinent=df_2021.copy()
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
country_data = list(dfContinent['Country name'].unique())
country_geo = list(world['name'])


# #### Searching for an different country names

# In[24]:


country_diff = [country for country in country_data if country not in country_geo]
country_diff


# #### Refactoring

# In[25]:


dfContinent['Country name'] = dfContinent['Country name'].replace({'United States' : 'United States of America','Taiwan Province of China':'Taiwan','Bosnia and Herzegovina':'Bosnia and Herz.','Dominican Republic':'Dominican Rep.', 'North Cyprus':'N. Cyprus',
'Swaziland':'Eswatini','Czech Republic' : 'Czechia','Congo (Brazzaville)':'Congo', 'Palestinian Territories' :'Palestine, State of',  'Hong Kong S.A.R. of China':'Hong Kong' })                                                                                    


# #### Jonining tables

# In[26]:


cont=world[['name','continent']]
cont.columns=['Country name', 'Continent']
dfW=dfContinent.merge(cont, on='Country name')


# In[27]:


dfTree=dfW.groupby(['Continent', 'Country name']).mean(['Healthy life expectancy']).reset_index()
dfTree['World']='World'

fig4=px.treemap(data_frame=dfTree, path=['World','Continent','Country name'], values='Happiness score')
fig4.update_layout(title=dict(text='World Happiness tree map', xanchor='center', yanchor='top', x=0.5),title_font_size=22)
fig4.show()


# ## Countries with higehest and  lowest happiness score

# In[28]:


dfhigh = df_2021.nlargest(10,"Happiness score")
dfhigh = dfhigh.sort_values(by=['Happiness score'], ascending=True)
dflow=  df_2021.nsmallest(10,"Happiness score")
dflow = dflow.sort_values(by=['Happiness score'], ascending=False)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Countries with highest happiness score','Countries with lowest happiness score'),
    horizontal_spacing = 0.12,
    vertical_spacing = 0.25)



fig.add_trace(go.Bar(
                y=dfhigh["Country name"],
                x=dfhigh["Happiness score"],
                hovertext=dfhigh["Happiness score"],
                orientation='h'),
                row=1, col=1)
fig.add_trace(go.Bar(
                y=dflow["Country name"],
                x=dflow["Happiness score"],
                hovertext=dflow["Happiness score"],
                orientation='h'),
                row=1, col=2)

                                                                                                        
fig.update_layout(title_text='<b>Countries with highest and lowest happiness score<b>', 
                  titlefont={'size':25},
                  title_x=0.5,
                  showlegend=False,
                  autosize=True,
                  height=1000,
                  template = "plotly_white",
                  yaxis_title_text='Country',
                  xaxis_title_text='Happiness score',
                 )

fig.show()




# ## Variable influence for countries with highest and lowest happiness score

# In[29]:


explained_featuresh = df_2021.filter(regex="Explained").columns.tolist()
explained_featuresh.append("Dystopia + residual")
my_list = []
for f,c in zip(explained_featuresh, px.colors.qualitative.D3):
    my_list.append(go.Bar(y=df_2021.nlargest(10,"Happiness score")["Country name"].values,
                          x=df_2021.nlargest(10,"Happiness score")[f].values,
                          name=f,
                          marker=dict(color=c),
                          orientation="h"))
fig = go.Figure(data=my_list)
fig.update_traces(marker_line_color='rgb(255,255,255)',
                  marker_line_width=2.5, opacity=0.7)

fig.update_layout(title='Variables influence of countries with highest happiness score',barmode='stack',title_font_size=22,margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000,legend_title_text='Variable')

fig.show()

explained_featuresl = df_2021.filter(regex="Explained").columns.tolist()
explained_featuresl.append("Dystopia + residual")
my_list = []
for f,c in zip(explained_featuresl, px.colors.qualitative.D3):
    my_list.append(go.Bar(y=df_2021.nsmallest(10,"Happiness score")["Country name"].values,
                          x=df_2021.nsmallest(10,"Happiness score")[f].values,
                          name=f,
                          marker=dict(color=c),
                          orientation="h"))
fig = go.Figure(data=my_list)
fig.update_traces(marker_line_color='rgb(255,255,255)',
                  marker_line_width=2.5, opacity=0.7)
fig.update_layout(title='Variables influence of countries with lowest happiness score',barmode='stack',title_font_size=22,margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000,legend_title_text='Variable')
fig.show()


# ### Variable influence in continental summaring

# In[30]:


explain_column=['Explained by: Log GDP per capita', 'Explained by: Social support',
       'Explained by: Healthy life expectancy',
       'Explained by: Freedom to make life choices',
       'Explained by: Generosity', 'Explained by: Perceptions of corruption',
       'Dystopia + residual']

col=explain_column.copy()
col.append('Continent')


# #### Average values for each continent

# In[31]:


dfContinent=dfW[col].groupby(['Continent']).mean().reset_index()


# #### Values scalling

# In[32]:


scalar=MinMaxScaler()
dfContinent[explain_column]=scalar.fit_transform(dfContinent[explain_column])


# In[33]:


def plot_polar(continent1,continent2):
    
    theta=dfContinent.columns[1:]
    r1= dfContinent[dfContinent['Continent']==continent1].iloc[:,1:].values.flatten().tolist()
    r2= dfContinent[dfContinent['Continent']==continent2].iloc[:,1:].values.flatten().tolist()


    g1=go.Scatterpolar(r = r1,theta = theta,fill = 'toself',name=continent1)
    g2=go.Scatterpolar(r = r2,theta = theta,fill = 'toself',name=continent2)
    
    data = [g1, g2]
    fig = go.Figure(data = data)
    fig.update_layout(title=dict(xanchor='center', yanchor='top', x=0.5),title_font_size=22)
    fig.show()


# #### Continents comparisons

# In[34]:


plot_polar('Europe','Asia')
plot_polar('North America','South America')
plot_polar('Europe','North America')
plot_polar('Asia','North America')
plot_polar('Europe','Africa')


# ### 2021 data correlation

# In[35]:


dfHeat=df_2021.copy()
dfHeat=dfHeat[['Happiness score','Logged GDP per capita', 'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption','Dystopia + residual']]
plt.figure(figsize=(25,12))
sub = np.triu(dfHeat.corr())
sns.heatmap(dfHeat.corr(), annot=True,linewidth=.15, mask=sub,cbar_kws={'label': 'Variables influence'})

plt.title('2021 data correlation')


# ### All of the variables impact the final happiness score

# ## 3.2 Data visualization with mapping

# ## 2021 data

# In[36]:


dfMap=df_2021.copy()
dfMap1 = dfMap.set_index('Country name')

ref = pd.DataFrame(dfMap1['Happiness score']).reset_index()
ref.loc[ref['Country name'] == 'Taiwan Province of China', 'Country name'] = 'Taiwan, Province of China' 
ref.loc[ref['Country name'] == 'Hong Kong S.A.R. of China', 'Country name'] = 'Hong Kong' 
ref.loc[ref['Country name'] == 'Congo (Brazzaville)','Country name'] = 'Congo' 
ref.loc[ref['Country name'] == 'Palestinian Territories','Country name'] = 'Palestine, State of' 
ref.drop(index=ref[ref['Country name'] == 'Kosovo'].index, inplace=True) 
ref.drop(index=ref[ref['Country name'] == 'North Cyprus'].index, inplace=True) 
ref['iso_alpha'] = ref['Country name'].apply(lambda x:pc.country_name_to_country_alpha3(x,))

fig = px.choropleth(ref, locations='iso_alpha',
                    color='Happiness score',
                    hover_name='Country name',
                    color_continuous_scale=px.colors.diverging.RdYlGn,
)

fig.update_layout(title=dict(text='World map - happiness score', xanchor='center', yanchor='top',x=0.5),title_font_size=22
    ,paper_bgcolor='rgb(248, 248, 255)',geo_bgcolor='rgb(248, 248, 255)',margin = dict(l=0,r=0,b=0,t=50,pad=0,autoexpand=True),autosize=False,width=1500,height=1000)

fig.show()


# In[37]:


dfMap2 = dfMap.set_index('Country name')

ref = pd.DataFrame(dfMap2['Social support']).reset_index()
ref.loc[ref['Country name'] == 'Taiwan Province of China', 'Country name'] = 'Taiwan, Province of China' 
ref.loc[ref['Country name'] == 'Hong Kong S.A.R. of China', 'Country name'] = 'Hong Kong' 
ref.loc[ref['Country name'] == 'Congo (Brazzaville)','Country name'] = 'Congo' 
ref.loc[ref['Country name'] == 'Palestinian Territories','Country name'] = 'Palestine, State of' 
ref.drop(index=ref[ref['Country name'] == 'Kosovo'].index, inplace=True) 
ref.drop(index=ref[ref['Country name'] == 'North Cyprus'].index, inplace=True) 
ref['iso_alpha'] = ref['Country name'].apply(lambda x:pc.country_name_to_country_alpha3(x,))

fig = px.choropleth(ref, locations='iso_alpha',
                    color='Social support',
                    hover_name='Country name',
                    color_continuous_scale=px.colors.diverging.RdYlGn,
)
fig.update_layout(title=dict(text='World map - social support', xanchor='center', yanchor='top',x=0.5),title_font_size=22
    ,paper_bgcolor='rgb(248, 248, 255)',geo_bgcolor='rgb(248, 248, 255)',margin = dict(l=0,r=0,b=0,t=50,pad=0,autoexpand=True),autosize=False,width=1500,height=1000)

fig.show()


# In[38]:


dfMap3 = dfMap.set_index('Country name')

ref = pd.DataFrame(dfMap3['Logged GDP per capita']).reset_index()
ref.loc[ref['Country name'] == 'Taiwan Province of China', 'Country name'] = 'Taiwan, Province of China' 
ref.loc[ref['Country name'] == 'Hong Kong S.A.R. of China', 'Country name'] = 'Hong Kong' 
ref.loc[ref['Country name'] == 'Congo (Brazzaville)','Country name'] = 'Congo' 
ref.loc[ref['Country name'] == 'Palestinian Territories','Country name'] = 'Palestine, State of' 
ref.drop(index=ref[ref['Country name'] == 'Kosovo'].index, inplace=True) 
ref.drop(index=ref[ref['Country name'] == 'North Cyprus'].index, inplace=True) 
ref['iso_alpha'] = ref['Country name'].apply(lambda x:pc.country_name_to_country_alpha3(x,))

fig = px.choropleth(ref, locations='iso_alpha',
                    color='Logged GDP per capita',
                    hover_name='Country name',
                    color_continuous_scale=px.colors.diverging.RdYlGn,
                   )
fig.update_layout(title=dict(text='World map - logged GDP per capita', xanchor='center', yanchor='top',x=0.5),title_font_size=22
    ,paper_bgcolor='rgb(248, 248, 255)',geo_bgcolor='rgb(248, 248, 255)',margin = dict(l=0,r=0,b=0,t=50,pad=0,autoexpand=True),autosize=False,width=1500,height=1000)
fig.show()


# In[39]:


dfMap4 = dfMap.set_index('Country name')

ref = pd.DataFrame(dfMap4['Healthy life expectancy']).reset_index()
ref.loc[ref['Country name'] == 'Taiwan Province of China', 'Country name'] = 'Taiwan, Province of China' 
ref.loc[ref['Country name'] == 'Hong Kong S.A.R. of China', 'Country name'] = 'Hong Kong' 
ref.loc[ref['Country name'] == 'Congo (Brazzaville)','Country name'] = 'Congo' 
ref.loc[ref['Country name'] == 'Palestinian Territories','Country name'] = 'Palestine, State of' 
ref.drop(index=ref[ref['Country name'] == 'Kosovo'].index, inplace=True) 
ref.drop(index=ref[ref['Country name'] == 'North Cyprus'].index, inplace=True) 
ref['iso_alpha'] = ref['Country name'].apply(lambda x:pc.country_name_to_country_alpha3(x,))

fig = px.choropleth(ref, locations='iso_alpha',
                    color='Healthy life expectancy',
                    hover_name='Country name',
                    color_continuous_scale=px.colors.diverging.RdYlGn,
)
fig.update_layout(title=dict(text='World map - healthy life expectancy', xanchor='center', yanchor='top',x=0.5),title_font_size=22
    ,paper_bgcolor='rgb(248, 248, 255)',geo_bgcolor='rgb(248, 248, 255)',margin = dict(l=0,r=0,b=0,t=50,pad=0,autoexpand=True),autosize=False,width=1500,height=1000)

fig.show()


# In[40]:


dfMap5 = dfMap.set_index('Country name')

ref = pd.DataFrame(dfMap5['Freedom to make life choices']).reset_index()
ref.loc[ref['Country name'] == 'Taiwan Province of China', 'Country name'] = 'Taiwan, Province of China' 
ref.loc[ref['Country name'] == 'Hong Kong S.A.R. of China', 'Country name'] = 'Hong Kong' 
ref.loc[ref['Country name'] == 'Congo (Brazzaville)','Country name'] = 'Congo' 
ref.loc[ref['Country name'] == 'Palestinian Territories','Country name'] = 'Palestine, State of' 
ref.drop(index=ref[ref['Country name'] == 'Kosovo'].index, inplace=True) 
ref.drop(index=ref[ref['Country name'] == 'North Cyprus'].index, inplace=True) 
ref['iso_alpha'] = ref['Country name'].apply(lambda x:pc.country_name_to_country_alpha3(x,))

fig = px.choropleth(ref, locations='iso_alpha',
                    color='Freedom to make life choices',
                    hover_name='Country name',
                    color_continuous_scale=px.colors.diverging.RdYlGn,
)
fig.update_layout(title=dict(text='World map - freedom to make life choices', xanchor='center', yanchor='top',x=0.5),title_font_size=22
    ,paper_bgcolor='rgb(248, 248, 255)',geo_bgcolor='rgb(248, 248, 255)',margin = dict(l=0,r=0,b=0,t=50,pad=0,autoexpand=True),autosize=False,width=1500,height=1000)

fig.show()


# In[41]:


dfMap6 = dfMap.set_index('Country name')

ref = pd.DataFrame(dfMap6['Generosity']).reset_index()
ref.loc[ref['Country name'] == 'Taiwan Province of China', 'Country name'] = 'Taiwan, Province of China' 
ref.loc[ref['Country name'] == 'Hong Kong S.A.R. of China', 'Country name'] = 'Hong Kong' 
ref.loc[ref['Country name'] == 'Congo (Brazzaville)','Country name'] = 'Congo' 
ref.loc[ref['Country name'] == 'Palestinian Territories','Country name'] = 'Palestine, State of' 
ref.drop(index=ref[ref['Country name'] == 'Kosovo'].index, inplace=True) 
ref.drop(index=ref[ref['Country name'] == 'North Cyprus'].index, inplace=True) 
ref['iso_alpha'] = ref['Country name'].apply(lambda x:pc.country_name_to_country_alpha3(x,))

fig = px.choropleth(ref, locations='iso_alpha',
                    color='Generosity',
                    hover_name='Country name',
                    color_continuous_scale=px.colors.diverging.RdYlGn,
)
fig.update_layout(title=dict(text='World map - generosity', xanchor='center', yanchor='top',x=0.5),title_font_size=22
    ,paper_bgcolor='rgb(248, 248, 255)',geo_bgcolor='rgb(248, 248, 255)',margin = dict(l=0,r=0,b=0,t=50,pad=0,autoexpand=True),autosize=False,width=1500,height=1000)

fig.show()


# In[42]:


dfMap7 = dfMap.set_index('Country name')

ref = pd.DataFrame(dfMap7['Perceptions of corruption']).reset_index()
ref.loc[ref['Country name'] == 'Taiwan Province of China', 'Country name'] = 'Taiwan, Province of China' 
ref.loc[ref['Country name'] == 'Hong Kong S.A.R. of China', 'Country name'] = 'Hong Kong' 
ref.loc[ref['Country name'] == 'Congo (Brazzaville)','Country name'] = 'Congo' 
ref.loc[ref['Country name'] == 'Palestinian Territories','Country name'] = 'Palestine, State of' 
ref.drop(index=ref[ref['Country name'] == 'Kosovo'].index, inplace=True) 
ref.drop(index=ref[ref['Country name'] == 'North Cyprus'].index, inplace=True) 
ref['iso_alpha'] = ref['Country name'].apply(lambda x:pc.country_name_to_country_alpha3(x,))

fig = px.choropleth(ref, locations='iso_alpha',
                    color='Perceptions of corruption',
                    hover_name='Country name',
                    color_continuous_scale=px.colors.diverging.RdYlGn,
)
fig.update_layout(title=dict(text='World map - perceptions of corruption', xanchor='center', yanchor='top',x=0.5),title_font_size=22
    ,paper_bgcolor='rgb(248, 248, 255)',geo_bgcolor='rgb(248, 248, 255)',margin = dict(l=0,r=0,b=0,t=50,pad=0,autoexpand=True),autosize=False,width=1500,height=1000)

fig.show()


# In[43]:


dfMap=df_2021.copy()
dfMap1 = dfMap.set_index('Country name')

ref = pd.DataFrame(dfMap1['Dystopia + residual']).reset_index()
ref.loc[ref['Country name'] == 'Taiwan Province of China', 'Country name'] = 'Taiwan, Province of China' 
ref.loc[ref['Country name'] == 'Hong Kong S.A.R. of China', 'Country name'] = 'Hong Kong' 
ref.loc[ref['Country name'] == 'Congo (Brazzaville)','Country name'] = 'Congo' 
ref.loc[ref['Country name'] == 'Palestinian Territories','Country name'] = 'Palestine, State of' 
ref.drop(index=ref[ref['Country name'] == 'Kosovo'].index, inplace=True) 
ref.drop(index=ref[ref['Country name'] == 'North Cyprus'].index, inplace=True) 
ref['iso_alpha'] = ref['Country name'].apply(lambda x:pc.country_name_to_country_alpha3(x,))

fig = px.choropleth(ref, locations='iso_alpha',
                    color='Dystopia + residual',
                    hover_name='Country name',
                    color_continuous_scale=px.colors.diverging.RdYlGn,
)

fig.update_layout(title=dict(text='World map - Dystopia + residual', xanchor='center', yanchor='top',x=0.5),title_font_size=22
    ,paper_bgcolor='rgb(248, 248, 255)',geo_bgcolor='rgb(248, 248, 255)',margin = dict(l=0,r=0,b=0,t=50,pad=0,autoexpand=True),autosize=False,width=1500,height=1000)
    
fig.show()


# ## Folium Maps

# ### Applying coordinates based on location

# In[44]:


dfFoliummMap=df_2021.copy()
geolocator = Nominatim(user_agent="User")
def geolocate(country):
    try:
        loc = geolocator.geocode(country)
        return (loc.latitude, loc.longitude)
    except:
        return np.nan
dfFoliummMap['lat_long']=dfFoliummMap['Country name'].apply(geolocate)
dfFoliummMap.head()


# ### Manual applying of coordinates for countries without match

# In[45]:


dfFoliummMap['lat_long']=np.where(dfFoliummMap['Country name']=='Hong Kong S.A.R. of China','(22.3193,114.1694)',dfFoliummMap['lat_long'])
dfFoliummMap['lat_long']=np.where(dfFoliummMap['Country name']=='Taiwan Province of China','(25.047,121.5318)',dfFoliummMap['lat_long'])


# In[46]:


dfFoliummMap[dfFoliummMap['lat_long'].isna()]


# ### Data transforming

# In[47]:


dfFoliummMap['lat_long']=dfFoliummMap['lat_long'].astype(str)
dfFoliummMap['lat_long']=dfFoliummMap['lat_long'].str.replace('(','')
dfFoliummMap['lat_long']=dfFoliummMap['lat_long'].str.replace(')','')
dfFoliummMap=pd.concat([dfFoliummMap,dfFoliummMap['lat_long'].str.split(',',expand=True).rename({0:'lat',1:'long'},axis=1)],axis=1)


# ### Visualization based on countries coordinates using Folium

# In[48]:


world_map= folium.Map(location=[45.3656,9.1651],zoom_start=3,max_bounds=True,no_wrap=True,tiles="cartodbpositron")
marker_cluster = MarkerCluster().add_to(world_map)
for i in range(len(dfFoliummMap)):
        lat = dfFoliummMap.iloc[i]['lat']
        long = dfFoliummMap.iloc[i]['long']
        radius=5
        popup_text = """Kraj : {}<br> Happiness score : {}"""  
        popup_text = popup_text.format(dfFoliummMap.iloc[i]['Country name'],dfFoliummMap.iloc[i]['Happiness score'],dfFoliummMap.iloc[i]['Logged GDP per capita'])
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True).add_to(marker_cluster)
world_map


# ## Historical data

# In[49]:


df_hist= pd.read_csv('data/HistoricalData.csv',sep=';',decimal=',')


# In[50]:


df_hist.head()


# ### Interactive historical maps

# In[51]:


dfIMAP=df_hist.copy()
dfIMAP.rename({"Life Ladder": "Happiness score"},axis=1, inplace=True)
df_choropleth=dfIMAP.groupby(['year','Country name']).mean(['Happiness score']).reset_index()
dfIMAP = dfIMAP.sort_values(['year'],ascending=[True])
fig = px.choropleth(data_frame=dfIMAP, locations='Country name',locationmode="country names", 
                    color='Happiness score',animation_frame='year',
                    color_continuous_scale=[(0, "red"), (0.5, "white"), (1, "blue")])

fig.update_layout(title=dict(text='Happiness score 2005-2021', xanchor='center', yanchor='top',x=0.5),title_font_size=22,margin = dict(l=0,r=0,b=0,t=50,pad=0,autoexpand=True),autosize=False,width=1500,height=1000)
fig.show()


# In[52]:


df_choropleth=dfIMAP.groupby(['year','Country name']).mean(['Log GDP per capita']).reset_index()

fig = px.choropleth(data_frame=dfIMAP, locations='Country name',locationmode="country names", 
                    color='Log GDP per capita',animation_frame='year',
                    color_continuous_scale=[(0, "red"), (0.5, "white"), (1, "blue")]
                   )

fig.update_layout(title=dict(text='Log GDP per capita 2005-2021', xanchor='center', yanchor='top',x=0.5),title_font_size=22,margin = dict(l=0,r=0,b=0,t=50,pad=0,autoexpand=True),autosize=False,width=1500,height=1000)
fig.show()


# In[53]:


df_choropleth=dfIMAP.groupby(['year','Country name']).mean(['Social support']).reset_index()

fig = px.choropleth(data_frame=dfIMAP, locations='Country name',locationmode="country names", 
                    color='Social support',animation_frame='year',
                    color_continuous_scale=[(0, "red"), (0.5, "white"), (1, "blue")])

fig.update_layout(title=dict(text='Social support 2005-2021', xanchor='center', yanchor='top',x=0.5),title_font_size=22,margin = dict(l=0,r=0,b=0,t=50,pad=0,autoexpand=True),autosize=False,width=1500,height=1000)
fig.show()


# In[54]:


df_choropleth=dfIMAP.groupby(['year','Country name']).mean(['Healthy life expectancy at birth']).reset_index()

fig = px.choropleth(data_frame=dfIMAP, locations='Country name',locationmode="country names", 
                    color='Healthy life expectancy at birth',animation_frame='year',
                    color_continuous_scale=[(0, "red"), (0.5, "white"), (1, "blue")])

fig.update_layout(title=dict(text='Healthy life expectancy at birth 2005-2021', xanchor='center', yanchor='top',x=0.5),title_font_size=22,margin = dict(l=0,r=0,b=0,t=50,pad=0,autoexpand=True),autosize=False,width=1500,height=1000)
fig.show()


# In[55]:


df_choropleth=dfIMAP.groupby(['year','Country name']).mean(['Freedom to make life choices']).reset_index()

fig = px.choropleth(data_frame=dfIMAP, locations='Country name',locationmode="country names", 
                    color='Freedom to make life choices',animation_frame='year',
                    color_continuous_scale=[(0, "red"), (0.5, "white"), (1, "blue")])

fig.update_layout(title=dict(text='Freedom to make life choices 2005-2021', xanchor='center', yanchor='top',x=0.5),title_font_size=22,margin = dict(l=0,r=0,b=0,t=50,pad=0,autoexpand=True),autosize=False,width=1500,height=1000)
fig.show()


# In[56]:


df_choropleth=dfIMAP.groupby(['year','Country name']).mean(['Generosity']).reset_index()

fig = px.choropleth(data_frame=dfIMAP, locations='Country name',locationmode="country names", 
                    color='Generosity',animation_frame='year',
                    color_continuous_scale=[(0, "red"), (0.5, "white"), (1, "blue")])

fig.update_layout(title=dict(text='Generosity 2005-2021', xanchor='center', yanchor='top',x=0.5),title_font_size=22,margin = dict(l=0,r=0,b=0,t=50,pad=0,autoexpand=True),autosize=False,width=1500,height=1000)
fig.show()


# In[57]:


df_choropleth=dfIMAP.groupby(['year','Country name']).mean(['Perceptions of corruption']).reset_index()

fig = px.choropleth(data_frame=dfIMAP, locations='Country name',locationmode="country names", 
                    color='Perceptions of corruption',animation_frame='year',
                    color_continuous_scale=[(0, "red"), (0.5, "white"), (1, "blue")])

fig.update_layout(title=dict(text='Perceptions of corruption 2005-2021', xanchor='center', yanchor='top',x=0.5),title_font_size=22,margin = dict(l=0,r=0,b=0,t=50,pad=0,autoexpand=True),autosize=False,width=1500,height=1000)
      
fig.show()


# # 4. Machine learning

# ## 4.1 Supervised learning

# ### Data preprocessing&transforming

# In[58]:


train=df_2021.copy()
train=train[['Happiness score','Logged GDP per capita', 'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption','Dystopia + residual']]
train.info()


# In[59]:


y=train['Happiness score']
x=train.drop(['Happiness score'],axis=1)


# In[60]:


X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=1)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[61]:


df_models = pd.DataFrame(data=None, columns=['Algorithm', 'MSE', 'MAE','R2'])

def make_model(X_tr, X_te, y_tr, y_te, model, model_name: str): 
    model.fit(X_tr, y_tr)
    y_pred=model.predict(X_te)
    MSE=mean_squared_error(y_te,y_pred)
    MAE=mean_absolute_error(y_te, y_pred)
    R2 = r2_score(y_te, y_pred)
    df_models.loc[len(df_models.index)] = [model_name, MSE, MAE,R2]


# In[62]:


models = [RandomForestRegressor(),DecisionTreeRegressor(),ElasticNet(), KNeighborsRegressor(), XGBRegressor(),Ridge(),LinearRegression(),linear_model.BayesianRidge(),SVR()]


# In[63]:


for model in models:
    make_model(X_train, X_test, y_train, y_test, model, f'{model}'[:10])


# In[64]:


fig = go.Figure(data=[
    go.Bar(name='MSE', x=df_models.Algorithm, y=df_models.MSE),
    go.Bar(name='MAE', x=df_models.Algorithm, y=df_models.MAE),
    go.Bar(name='R2', x=df_models.Algorithm, y=df_models.R2)
])

fig.update_layout(title='MAE,MSE,R2 of happiness score for different regression models',title_font_size=22,margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000,legend_title_text='Legend')


# ## 4.1.2 Comparing variables influence for different regression models

# ###  Data transforming

# In[65]:


df5=df_2021.copy()
y=df5['Happiness score']
X=df5[['Logged GDP per capita', 'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption','Dystopia + residual']]


# In[66]:


X_train,X_test,Y_train,Y_test= train_test_split(X,y,test_size=0.2,random_state=1)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ### Linear Regression

# In[67]:


lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train)
y_hat = lin_reg.predict(X_test)
R2_train = lin_reg.score(X_train, Y_train)
R2_test = lin_reg.score(X_test, Y_test)
mse = mean_squared_error(Y_test, y_hat)
R2_test, mse


# In[68]:


coefficients = pd.DataFrame({"Feature":X.columns,"Coefficients":np.transpose(lin_reg.coef_)})
fig = px.bar(coefficients, x="Feature", y="Coefficients")
fig.update_layout(title_text='<b>Linear Regression<b>', 
                  titlefont={'size':25},
                  title_x=0.5,
                  showlegend=False,
                  autosize=True,
                  height=1000,
                  template = "plotly_white",
                  yaxis_title_text='Variable influence',
                  xaxis_title_text='Variable',
                 )


# ## Ridge Regression

# In[69]:


RRmodel = Ridge()
RRmodel.fit(X_train, Y_train)
y_hat = RRmodel.predict(X_test)
R2_train = RRmodel.score(X_train, Y_train)
R2_test = RRmodel.score(X_test, Y_test)
mse = mean_squared_error(Y_test, y_hat)
R2_test, mse


# In[70]:


coefficients = pd.DataFrame({"Feature":X.columns,"Coefficients":np.transpose(RRmodel.coef_)})
fig = px.bar(coefficients, x="Feature", y="Coefficients")
fig.update_layout(title_text='<b>Ridge Regression<b>', 
                  titlefont={'size':25},
                  title_x=0.5,
                  showlegend=False,
                  autosize=True,
                  height=1000,
                   template = "plotly_white",
                  yaxis_title_text='Variable influence',
                  xaxis_title_text='Variable',
                 )


# ### ElasticNet Regression

# In[71]:


ENmodel = ElasticNet()
ENmodel.fit(X_train, Y_train)
y_hat = ENmodel.predict(X_test)
R2_train = ENmodel.score(X_train, Y_train)
R2_test = ENmodel.score(X_test, Y_test)
mse = mean_squared_error(Y_test, y_hat)
R2_test, mse


# In[72]:


coefficients = pd.DataFrame({"Feature":X.columns,"Coefficients":np.transpose(ENmodel.coef_)})
fig = px.bar(coefficients, x="Feature", y="Coefficients")
fig.update_layout(title_text='<b>Elastic Net Regression<b>', 
                  titlefont={'size':25},
                  title_x=0.5,
                  showlegend=False,
                  autosize=True,
                  height=1000,
                   template = "plotly_white",
                  yaxis_title_text='Variable influence',
                  xaxis_title_text='Variable',
                 )


# ### SVR Regression

# In[73]:


svr = SVR(kernel='linear')
svr.fit(X_train, Y_train)
y_hat = svr.predict(X_test)
R2_train = svr.score(X_train, Y_train)
R2_test = svr.score(X_test, Y_test)
mse = mean_squared_error(Y_test, y_hat)
R2_test, mse


# In[74]:


coefficients = pd.DataFrame({"Feature":X.columns,"Coefficients":np.transpose(svr.coef_[0])})
fig = px.bar(coefficients, x="Feature", y="Coefficients")
fig.update_layout(title_text='<b>SVR Regression<b>', 
                  titlefont={'size':25},
                  title_x=0.5,
                  showlegend=False,
                  autosize=True,
                  height=1000,
                   template = "plotly_white",
                  yaxis_title_text='Variable influence',
                  xaxis_title_text='Variable',
                 )


# ### BayesianRidge Regression

# In[75]:


BRmodel = linear_model.BayesianRidge()
BRmodel.fit(X_train, Y_train)
y_hat = BRmodel.predict(X_test)
R2_train = BRmodel.score(X_train, Y_train)
R2_test = BRmodel.score(X_test, Y_test)
mse = mean_squared_error(Y_test, y_hat)
R2_test, mse


# In[76]:


coefficients = pd.DataFrame({"Feature":X.columns,"Coefficients":np.transpose(BRmodel.coef_)})
fig = px.bar(coefficients, x="Feature", y="Coefficients")
fig.update_layout(title_text='<b>Bayesian Ridge Regression<b>', 
                  titlefont={'size':25},
                  title_x=0.5,
                  showlegend=False,
                  autosize=True,
                  height=1000,
                  template = "plotly_white",
                  yaxis_title_text='Variable influence',
                  xaxis_title_text='Variable',
                 )


# ### DecisionTree Regression

# In[77]:


dtr= DecisionTreeRegressor()
dtr.fit(X_train,Y_train)
y_pred = dtr.predict(X_test)
test_mse = mean_squared_error(Y_test, y_pred)
y_pred_train = dtr.predict(X_train)
train_mse = mean_squared_error(Y_train, y_pred_train)
dtr.score(X_test, Y_test), test_mse 


# In[78]:


coefficients = pd.DataFrame({"Feature":X.columns,"Coefficients":dtr.feature_importances_})
fig = px.bar(coefficients, x="Feature", y="Coefficients")
fig.update_layout(title_text='<b>Decision Tree Regression<b>', 
                  titlefont={'size':25},
                  title_x=0.5,
                  showlegend=False,
                  autosize=True,
                  height=1000,
                   template = "plotly_white",
                  yaxis_title_text='Variable influence',
                  xaxis_title_text='Variable',
                 )


# ### Random Forest Regression

# In[79]:


rf = RandomForestRegressor()
rf.fit(X_train, Y_train)
y_hat = rf.predict(X_test)
errors = abs(y_hat - Y_test)
acc = 1 - errors
rf.score(X_test, Y_test), np.mean(acc)


# In[80]:


coefficients = pd.DataFrame({"Feature":X.columns,"Coefficients":rf.feature_importances_})
fig = px.bar(coefficients, x="Feature", y="Coefficients")
fig.update_layout(title_text='<b>Random Forest Regression<b>', 
                  titlefont={'size':25},
                  title_x=0.5,
                  showlegend=False,
                  autosize=True,
                  height=1000,
                   template = "plotly_white",
                  yaxis_title_text='Variable influence',
                  xaxis_title_text='Variable',
                 )


# ### XGB Regression

# In[81]:


xgbR = XGBRegressor()
xgbR.fit(X_train, Y_train)
y_hat = xgbR.predict(X_test)
R2_train = xgbR.score(X_train, Y_train)
R2_test = xgbR.score(X_test, Y_test)
mse = mean_squared_error(Y_test, y_hat)
R2_test, mse


# In[82]:


coefficients = pd.DataFrame({"Feature":X.columns,"Coefficients":xgbR.feature_importances_})
fig = px.bar(coefficients, x="Feature", y="Coefficients")
fig.update_layout(title_text='<b>XGB Regression<b>', 
                  titlefont={'size':25},
                  title_x=0.5,
                  showlegend=False,
                  autosize=True,
                  height=1000,
                   template = "plotly_white",
                  yaxis_title_text='Variable influence',
                  xaxis_title_text='Variable',
                 )


# ## 4.2 Unsupervised learning (clustering)

# ## Data normalization

# In[83]:


dfCluster=df_2021.copy()
dfCluster.drop(['lowerwhisker','upperwhisker','Happiness score in Dystopia','Standard error of ladder score','Explained by: Log GDP per capita','Explained by: Social support','Explained by: Healthy life expectancy','Explained by: Freedom to make life choices','Explained by: Generosity','Explained by: Perceptions of corruption'],axis=1,inplace=True)
country=dfCluster[dfCluster.columns[0]]
dataC= dfCluster.iloc[:,2:]


# In[84]:


def normalizedData(x):
    normalised = StandardScaler()
    normalised.fit_transform(x)
    return(x)


# In[85]:


dataC = normalizedData(dataC)


# ## K-means clustering

# In[86]:


def Kmeans(x, y):
    km= cluster.KMeans(x)
    km_result=km.fit_predict(y)
    return(km_result)


# ### Cluster groups

# In[87]:


km_result = Kmeans(8,dataC)
dataC['Kmeans_Cluster'] = pd.DataFrame(km_result)
dataset=pd.concat([dataC,country],axis=1)
dataset.groupby(by=["Kmeans_Cluster"]).mean()


# In[88]:


dataPlot = dict(type = 'choropleth', 
           locations = dataset['Country name'],
           locationmode = 'country names',
           z = dataset['Kmeans_Cluster'], 
           text = dataset['Country name'],
           colorbar = {'title':'Cluster group'})
layout = dict(title = 'K-means', 
           geo = dict(showframe = False, ), autosize=False,title_font_size=22,
        margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000)
choromap3 = go.Figure(data = [dataPlot], layout=layout)
iplot(choromap3) 


# ## Minibatch Kmeans clustering

# In[89]:


def MiniKmeans(x, y):
    mb= cluster.MiniBatchKMeans(x)
    mb_result=mb.fit_predict(y)
    return(mb_result)


# ### Cluster groups

# In[90]:


dataC.drop(['Kmeans_Cluster'],axis=1,inplace=True)
mb_result = MiniKmeans(8,dataC)
dataC['MiniKmeans_Cluster'] = pd.DataFrame(mb_result)
dataset=pd.concat([dataC,country],axis=1)
dataset.groupby(by=["MiniKmeans_Cluster"]).mean()


# In[91]:


dataPlot = dict(type = 'choropleth', 
           locations = dataset['Country name'],
           locationmode = 'country names',
           z = dataset['MiniKmeans_Cluster'], 
           text = dataset['Country name'],
           colorbar = {'title':'Cluster group'})
layout = dict(title = 'Minibatch K-means', 
           geo = dict(showframe = False, )
             , autosize=False,title_font_size=22,
        margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000)
choromap3 = go.Figure(data = [dataPlot], layout=layout)
iplot(choromap3) 


# ## DBSCAN clustering

# In[92]:


def Dbscan(x, y):
    db=cluster.DBSCAN(eps=x)
    db_result=db.fit_predict(y)
    return(db_result)


# ### Cluster groups

# In[93]:


dataC.drop(['MiniKmeans_Cluster'],axis=1,inplace=True)
db_result = Dbscan(0.95,dataC)
dataC['Dbscan_cluster'] = pd.DataFrame(db_result)
dataset=pd.concat([dataC,country],axis=1)
dataset.groupby(by=["Dbscan_cluster"]).mean()


# In[94]:


dataPlot = dict(type = 'choropleth', 
           locations = dataset['Country name'],
           locationmode = 'country names',
           z = dataset['Dbscan_cluster'], 
           text = dataset['Country name'],
           colorbar = {'title':'Cluster group'})
layout = dict(title = 'DBSCAN', 
           geo = dict(showframe = False, ), autosize=False,title_font_size=22,
        margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000)
choromap3 = go.Figure(data = [dataPlot], layout=layout)
iplot(choromap3) 


# ## Affinity Propagation clustering

# In[95]:


def Affinity(x, y,z):
    ap=cluster.AffinityPropagation(damping=x, preference=y)
    ap_result=ap.fit_predict(z)
    return(ap_result)


# ### Cluster groups

# In[96]:


dataC.drop(['Dbscan_cluster'],axis=1,inplace=True)
ap_result = Affinity(0.9,-100,dataC)
dataC['Affinity_Cluster'] = pd.DataFrame(ap_result)
dataset=pd.concat([dataC,country],axis=1)
dataset.groupby(by=["Affinity_Cluster"]).mean()


# In[97]:


dataPlot = dict(type = 'choropleth', 
           locations = dataset['Country name'],
           locationmode = 'country names',
           z = dataset['Affinity_Cluster'], 
           text = dataset['Country name'],
           colorbar = {'title':'Cluster group'})
layout = dict(title = 'Affinity Propagation', 
           geo = dict(showframe = False, ), autosize=False,title_font_size=22,
        margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000)
choromap3 = go.Figure(data = [dataPlot], layout=layout)
iplot(choromap3) 


# ##  Gaussian Mixture clustering

# In[98]:


def gmm(x, y):
    gm=mixture.GaussianMixture(n_components=x,covariance_type='full')
    gm.fit(y)
    gm_result=gm.predict(y)
    return(gm_result)


# ### Cluster groups

# In[99]:


dataC.drop(['Affinity_Cluster'],axis=1,inplace=True)
gm_result = gmm(8,dataC)
dataC['GM_Cluster'] = pd.DataFrame(gm_result)
dataset=pd.concat([dataC,country],axis=1)
dataset.groupby(by=["GM_Cluster"]).mean()


# In[100]:


dataPlot = dict(type = 'choropleth', 
           locations = dataset['Country name'],
           locationmode = 'country names',
           z = dataset['GM_Cluster'], 
           text = dataset['Country name'],
           colorbar = {'title':'Cluster group'})
layout = dict(title = 'Gaussian Mixture', 
           geo = dict(showframe = False, ), autosize=False,title_font_size=22,
        margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000)
choromap3 = go.Figure(data = [dataPlot], layout=layout)
iplot(choromap3) 


# ## BIRCH clustering

# In[101]:


def Bir(x, y):
    bi=cluster.Birch(n_clusters=x)
    bi_result=bi.fit_predict(y)
    return(bi_result)


# ### Cluster groups

# In[102]:


dataC.drop(['GM_Cluster'],axis=1,inplace=True)
bi_result = Bir(8,dataC)
dataC['Birch_Cluster'] = pd.DataFrame(bi_result)
dataset=pd.concat([dataC,country],axis=1)
dataset.groupby(by=["Birch_Cluster"]).mean()


# In[103]:


dataPlot = dict(type = 'choropleth', 
           locations = dataset['Country name'],
           locationmode = 'country names',
           z = dataset['Birch_Cluster'], 
           text = dataset['Country name'],
           colorbar = {'title':'Cluster group'})
layout = dict(title = 'BIRCH', 
           geo = dict(showframe = False, ), autosize=False,title_font_size=22,
        margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000)
choromap3 = go.Figure(data = [dataPlot], layout=layout)
iplot(choromap3) 


# ## Spectral clustering

# In[104]:


n_clusters=8 
spectral = cluster.SpectralClustering(n_clusters=n_clusters)
sp_result= spectral.fit_predict(dataC)


# ### Cluster groups

# In[105]:


dataC.drop(['Birch_Cluster'],axis=1,inplace=True)
dataC['Spectral_Cluster'] = pd.DataFrame(sp_result)
dataset=pd.concat([dataC,country],axis=1)
dataset.groupby(by=["Spectral_Cluster"]).mean()


# In[106]:


dataPlot = dict(type = 'choropleth', 
           locations = dataset['Country name'],
           locationmode = 'country names',
           z = dataset['Spectral_Cluster'], 
           text = dataset['Country name'],
           colorbar = {'title':'Cluster group'})
layout = dict(title = 'Spectral', 
           geo = dict(showframe = False, ), autosize=False,title_font_size=22,
        margin = dict(l=0,r=0,b=0,t=40,pad=4,autoexpand=True),
            width=1500,height=1000)
choromap3 = go.Figure(data = [dataPlot], layout=layout)
iplot(choromap3) 

