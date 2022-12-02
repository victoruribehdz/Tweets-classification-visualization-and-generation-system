# credenciales

bearer_token = "AAAAAAAAAAAAAAAAAAAAAAF5iAEAAAAAKdAqKfzJn1eKzu%2BPXUJG9IHx%2F60%3DM7yeueA4BUXJBuCDUqBNfgAF77flX3Ep0peN5uXVTdMY2Cp82P"

import os
import json
import datetime
from pymongo import MongoClient
import pandas as pd
from pandas import json_normalize
import streamlit as st
import pymongo
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from bokeh.plotting import figure 
import plotly.express as px 
#nltk.download('punkt')

#connection with mongodb atlas
ATLAS_URI='mongodb+srv://tuiter:tuiter@cluster0.avnamve.mongodb.net/?retryWrites=true&w=majority'
DB_NAME="TwitterStream"
COLLECTION_NAME="tweets"

# #connection with mongodb local
# MONGO_URI = os.environ.get('MONGO_URI')
# DB_NAME = os.environ.get('DB_NAME')
# COLLECTION_NAME = os.environ.get('COLLECTION_NAME')

#connection with mongodb atlas
client = MongoClient(ATLAS_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


# db.tweets.ensure_index("id", unique=True, dropDups=True)
# collection = db.tweets



    # rerun the app

    # Initialize connection.
    # Uses st.experimental_singleton to only run once.
@st.experimental_singleton
def get_connection():
    client = pymongo.MongoClient(ATLAS_URI)
    return client[DB_NAME][COLLECTION_NAME]

    # Get the connection.
connection = get_connection()

    # Pull data from the collection.
    # Uses st.experimental_memo to only rerun when the query changes or after 15 seconds.
@st.experimental_memo(ttl=15)
# title

def get_data():
    return list(connection.find())
    
# title
st.title("House of Dragons vs The Ring of Power")
items = get_data()
    #st.write(items)
    # get the number of tweets
if len(items) > 0:
    no_tweets = len(items)

        # get the number of unique users
    users = set()
    for item in items:
        users.add(item['username '])
    no_users = len(users)

        # get the number of tweets per user
    tweets_per_user = {}
    for item in items:
        user = item['username ']
        if user in tweets_per_user:
            tweets_per_user[user] += 1
        else:
            tweets_per_user[user] = 1
        #sum all replays
        #replays = 0
        #for item in items:
            #replays += item['replays']
        #sum all likes
    likes = 0
    for item in items:
        likes += item['likes']
        #sum all retweets
    retweets = 0
    for item in items:
        retweets += item['retweets']

        # get the highest number of tweets per user
    max_tweets = max(tweets_per_user.values())
        # get the user with the highest number of tweets
    max_user = [user for user, tweets in tweets_per_user.items() if tweets == max_tweets][0]

else:
    no_tweets = 0
    no_users = 0
    max_tweets = 0
    max_user = None
    replays = 0
    likes = 0


    # display the results
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Tweets", no_tweets)
        #reply
        #st.metric("Total Replies", replays)
with col2:
    st.metric("Total Users", no_users)
        #retweet
    st.metric("Total Retweets", retweets)
with col3:
    st.metric("Maximum # of Tweets", max_tweets, f"by {max_user}")
        #likes
    st.metric("Total Likes", likes)

df = json_normalize(collection.find())    
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

# df with show == got 
df_got = df[df['show'] == 'got']
df_got = df_got[['sentiment']]

# count by sentiment
df_got = df_got.groupby('sentiment').size().reset_index(name='counts')
# if sentiment == neutral change to neutro
df_got['sentiment'] = df_got['sentiment'].replace('neutro', 'neutral')

# df with show == rop
df_rop = df[df['show'] == 'rop']
df_rop = df_rop[['sentiment']]



# count by sentiment
df_rop = df_rop.groupby('sentiment').size().reset_index(name='counts')

# plot batchart
fig = go.Figure(data=[
    go.Bar(name='House of Dragons', x=df_got['sentiment'], y=df_got['counts']),
    go.Bar(name='The Ring of Power', x=df_rop['sentiment'], y=df_rop['counts'])
])
# Change the bar mode
fig.update_layout(barmode='group')
# delete background
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
# delete grid
fig.update_layout(showlegend=False)

fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
# add title
fig.update_layout(title_text='Sentiment Analysis')
#add legends
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

st.plotly_chart(fig)

# df2 with only data with show == 'got'
df2 = df[df['show'] == 'got']

# df3 with only data with show == 'rop'
df3 = df[df['show'] == 'rop']

# df2 keep date, sentiment

df2 = df2[['date', 'sentiment']]

# df3 keep date, sentiment

df3 = df3[['date', 'sentiment']]

# df2 group by datetime hour and sentiment

df2 = df2.groupby([pd.Grouper(key='date', freq='H'), 'sentiment']).size().reset_index(name='count')

# df3 group by datetime hour and sentiment
df3 = df3.groupby([pd.Grouper(key='date', freq='H'), 'sentiment']).size().reset_index(name='count')

# df2 pivot table
df2 = df2.pivot(index='date', columns='sentiment', values='count')

# df3 pivot table
df3 = df3.pivot(index='date', columns='sentiment', values='count')

# df2 fillna with 0
df2 = df2.fillna(0)

# df3 fillna with 0
df3 = df3.fillna(0)

# plot positive of df2 and df3
fig2 = px.line(df2, x=df2.index, y='positive', title='Positive tweets per hour')

fig2.add_scatter(x=df3.index, y=df3['positive'], mode='lines', name='positive Rop')

# delete background
fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)')
fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)')
# delete negative title
fig2.update_yaxes(title_text='')
fig2.update_xaxes(title_text='')
fig2.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

st.plotly_chart(fig2)

fig3 = px.line(df2, x=df2.index, y='neutro', title='Neutral tweets per hour')

fig3.add_scatter(x=df3.index, y=df3['neutral'], mode='lines', name='Neutral Rop')

# delete background
fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)')
fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)')
# delete negative title
fig3.update_yaxes(title_text='')
fig3.update_xaxes(title_text='')
fig3.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

st.plotly_chart(fig3)

# plot positive of df2 and df3
fig4 = px.line(df2, x=df2.index, y='negative', title='Negative tweets per hour')

fig4.add_scatter(x=df3.index, y=df3['negative'], mode='lines', name='Negative Rop')

# delete background
fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)')
fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)')
# delete negative title
fig4.update_yaxes(title_text='')
fig4.update_xaxes(title_text='')
fig4.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

st.plotly_chart(fig4)




#df_got and df_got in a pydec
df_got_lat_long = df[df['show'] == 'got']
df_got_lat_long = df_got_lat_long[['lat', 'long']]
df_rop_lat_long = df[df['show'] == 'rop']
df_rop_lat_long = df_rop_lat_long[['lat', 'long']]
df_got_lat_long = df_got_lat_long.dropna()
df_rop_lat_long = df_rop_lat_long.dropna()
df_got_lat_long = df_got_lat_long.drop_duplicates()
df_rop_lat_long = df_rop_lat_long.drop_duplicates()

#plot both in a single map
fig5 = px.scatter_mapbox(df_got_lat_long, lat="lat", lon="long", hover_name="lat", hover_data=["long"],
                    color_discrete_sequence=["fuchsia"], zoom=2, height=300)
fig5.add_scattermapbox(lat=df_rop_lat_long['lat'], lon=df_rop_lat_long['long'], mode='markers', marker=dict(size=5, color='blue', opacity=0.7))
fig5.update_layout(mapbox_style="open-street-map")
fig5.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# show usa
fig5.update_layout(mapbox_center_lon=-100)
fig5.update_layout(mapbox_center_lat=40)

st.plotly_chart(fig5)

