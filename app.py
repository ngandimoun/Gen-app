import streamlit as st
import plotly.express as px
import os
import openai
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np # linear algebra
import pandas as pd # for data preparation
import nltk
import PIL
import plotly.express as px # for data visualization
from textblob import TextBlob # for sentiment analysis
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from scipy import stats
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import string
import logging
logging.getLogger('openai').setLevel(logging.ERROR)

ima= PIL.Image.open('C:/Users/PC/Downloads/rosaline/assets/thumb.webp')
bon= PIL.Image.open('C:/Users/PC/Downloads/rosaline/assets/big.jpg')




# Display the top image in the sidebar
st.sidebar.image(ima, width=300)




# Create a function for each page of the app


st.write('CHAT-GPT3 FOR DATA SCIENCE', width=500)
dff=pd.read_csv('netflix_titles.csv')

dff.shape

#IPL

ba = pd.read_csv("Book_ipl22_ver_33.csv")



figo = px.bar(ba, x=["match_winner"],
            title="Number of Matches Won in IPL 2022")


ba["won_by"] = ba["won_by"].map({"Wickets": "Chasing", 
                                     "Runs": "Defending"})
won_by = ba["won_by"].value_counts()
label = won_by.index
counts = won_by.values
colors = ['gold','lightgreen']

figo1 = go.Figure(data=[go.Pie(labels=label, values=counts)])
figo1.update_layout(title_text='Number of Matches Won By Defending Or Chasing')
figo1.update_traces(hoverinfo='label+percent', textinfo='value', 
                  textfont_size=30,
                  marker=dict(colors=colors, 
                              line=dict(color='black', width=3)))



toss = ba["toss_decision"].value_counts()
label = toss.index
counts = toss.values
colors = ['skyblue','yellow']

figo2 = go.Figure(data=[go.Pie(labels=label, values=counts)])
figo2.update_layout(title_text='Toss Decision')
figo2.update_traces(hoverinfo='label+percent', 
                  textinfo='value', textfont_size=30,
                  marker=dict(colors=colors, 
                              line=dict(color='black', width=3)))



figo3 = px.bar(ba, x=ba["top_scorer"],
            title="Top Scorers in IPL 2022")



figo4 = px.bar(ba, x=ba["top_scorer"], 
                y = ba["highscore"], 
                color = ba["highscore"],
            title="Top Scorers in IPL 2022")



figo5 = px.bar(ba, x = ba["player_of_the_match"], 
                title="Most Player of the Match Awards")


figo6 = px.bar(ba, x=ba["best_bowling"],
            title="Best Bowlers in IPL 2022")



#Now let’s have a look at whether most of the wickets fall while setting the 
#target or while chasing the target:




figo7 = go.Figure()
figo7.add_trace(go.Bar(
    x=ba["venue"],
    y=ba["first_ings_wkts"],
    name='First Innings Wickets',
    marker_color='gold'
))
figo7.add_trace(go.Bar(
    x=ba["venue"],
    y=ba["second_ings_wkts"],
    name='Second Innings Wickets',
    marker_color='lightgreen'
))
figo7.update_layout(barmode='group', xaxis_tickangle=-45)




#apple

ap = pd.read_csv("apple_products.csv")


highest_rated = ap.sort_values(by=["Star Rating"], 
                             ascending=False)
highest_rated = highest_rated.head(10)
#print(highest_rated['Product Name'])

#Now let’s have a look at the number of ratings of the highest-rated iPhones on Flipkart:

iphones = highest_rated["Product Name"].value_counts()
label = iphones.index
counts = highest_rated["Number Of Ratings"]
apo = px.bar(highest_rated, x=label, 
                y = counts, 
            title="Number of Ratings of Highest Rated iPhones")

#Now let’s have a look at the number of reviews of the highest-rated iPhones on Flipkart:

iphones = highest_rated["Product Name"].value_counts()
label = iphones.index
counts = highest_rated["Number Of Reviews"]
apo1 = px.bar(highest_rated, x=label, 
                y = counts, 
            title="Number of Reviews of Highest Rated iPhones")


#Now let’s have a look at the relationship between the sale price of iPhones and their ratings on Flipkart:

apo3 = px.scatter(data_frame = ap, x="Number Of Ratings",
                    y="Sale Price", size="Discount Percentage", 
                    trendline="ols", 
                    title="Relationship between Sale Price and Number of Ratings of iPhones")


#Now let’s have a look at the relationship between the discount percentage on iPhones on Flipkart and the number of ratings:

apo4 = px.scatter(data_frame = ap, x="Number Of Ratings",
                    y="Discount Percentage", size="Sale Price", 
                    trendline="ols", 
                    title="Relationship between Discount Percentage and Number of Ratings of iPhones")






data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/water_potability.csv")



# there are no factors that we cannot ignore that affect water quality, so let’s explore all the columns one by one. Let’s start by looking at the ph column:

import plotly.express as px
data = data
figure = px.scatter(data, x="ph", y="Organic_carbon",
             color="Organic_carbon", color_continuous_scale='Inferno', title= "Factors Affecting Water Quality: PH")
#The ph column represents the ph value of the water which is an important factor in evaluating the acid-base balance of the water. 
#The pH value of drinking water should be between 6.5 and 8.5. 
#Now let’s look at the second factor affecting water quality in the dataset:

figure1 = px.scatter(data, x="Hardness", y="Organic_carbon",
             color="Organic_carbon", color_continuous_scale='Inferno', title= "Factors Affecting Water Quality: Hardness")
figure2 = px.scatter(data, x="Solids", y="Organic_carbon",
             color="Organic_carbon", color_continuous_scale='Inferno', title= "Factors Affecting Water Quality: Solids")



figure3 = px.scatter(data, x="Chloramines", y="Organic_carbon",
             color="Organic_carbon", color_continuous_scale='Inferno', title= "Factors Affecting Water Quality: Chloramines")


figure4 = px.scatter(data, x="Sulfate", y="Organic_carbon",
             color="Organic_carbon", color_continuous_scale='Inferno', title= "Factors Affecting Water Quality: Sulfate")


figure5 = px.scatter(data, x="Conductivity", y="Organic_carbon",
             color="Organic_carbon", color_continuous_scale='Inferno', title= "Factors Affecting Water Quality: Conductivity")


#TOP 5 DIRECTORS

dff['director']=dff['director'].fillna('No Director Specified')
filtered_directors=pd.DataFrame()
filtered_directors=dff['director'].str.split(',',expand=True).stack()
filtered_directors=filtered_directors.to_frame()
filtered_directors.columns=['Director']
directors=filtered_directors.groupby(['Director']).size().reset_index(name='Total Content')
directors=directors[directors.Director !='No Director Specified']
directors=directors.sort_values(by=['Total Content'],ascending=False)
directorsTop5=directors.head()
directorsTop5=directorsTop5.sort_values(by=['Total Content'])
fig1=px.bar(directorsTop5,x='Total Content',y='Director',title='Top 5 Directors on Netflix')


#TOP 5 ACTORS


dff['cast']=dff['cast'].fillna('No Cast Specified')
filtered_cast=pd.DataFrame()
filtered_cast=dff['cast'].str.split(',',expand=True).stack()
filtered_cast=filtered_cast.to_frame()
filtered_cast.columns=['Actor']
actors=filtered_cast.groupby(['Actor']).size().reset_index(name='Total Content')
actors=actors[actors.Actor !='No Cast Specified']
actors=actors.sort_values(by=['Total Content'],ascending=False)
actorsTop5=actors.head()
actorsTop5=actorsTop5.sort_values(by=['Total Content'])
fig2=px.bar(actorsTop5,x='Total Content',y='Actor', title='Top 5 Actors on Netflix')

#ANALYZING CONTENT ON NETFLIX


df1=dff[['type','release_year']]
df1=df1.rename(columns={"release_year": "Release Year"})
df2=df1.groupby(['Release Year','type']).size().reset_index(name='Total Content')
df2=df2[df2['Release Year']>=2010]
fig3 = px.line(df2, x="Release Year", y="Total Content", color='type',title='Trend of content produced over the years on Netflix')

#CONTENT SENTIMENT ANALYSIS

dfx=dff[['release_year','description']]
dfx=dfx.rename(columns={'release_year':'Release Year'})
for index,row in dfx.iterrows():
    z=row['description']
    testimonial=TextBlob(z)
    p=testimonial.sentiment.polarity
    if p==0:
        sent='Neutral'
    elif p>0:
        sent='Positive'
    else:
        sent='Negative'
    dfx.loc[[index,2],'Sentiment']=sent


dfx=dfx.groupby(['Release Year','Sentiment']).size().reset_index(name='Total Content')

dfx=dfx[dfx['Release Year']>=2010]
fig4 = px.bar(dfx, x="Release Year", y="Total Content", color="Sentiment", title="Sentiment of content on Netflix")
#fig4.show()


#Here is a way to analyze how the selection of content on Netflix has changed over time and 
#how this impacts the demand for different types of content using Python:
        
        #start
# Group the DataFrame by year and genre, and count the number of occurrences
counts = dff.groupby(['release_year', 'type'])['title'].count().reset_index(name='counts')


# Pivot the DataFrame to create a wide-form representation
pivot = counts.pivot(index='release_year', columns='type', values='counts').reset_index()

# Fill missing values with 0
pivot = pivot.fillna(0)

# Plot the results using a stacked bar chart
fig = px.bar(pivot, x='release_year', y=pivot.columns[1:], title='Number of Movies by Genre and Year on Netflix')
#fig.show()

# Visualize the relationship between the popularity of movies or TV shows and their release year
sns.scatterplot(x='release_year', y='rating', data=dff)

fig9 = px.bar(dff, x="release_year", y="rating", color='rating')



# Load the Netflix dataset
netflix = pd.read_csv('netflix_titles.csv')


# Load the Amazon Prime dataset
amazon_prime = pd.read_csv('amazon_prime_titles.csv')


# Load the Hulu dataset
hulu = pd.read_csv('hulu_titles.csv')


# Load the Disney+ dataset
disney_plus = pd.read_csv('disney_plus_titles.csv')


#Compare the distribution of content types across the different streaming platforms:

# Create a bar chart showing the number of movies, TV shows, and other types of content on each streaming platform
fig, ax = plt.subplots()

# Group the datasets by content type and count the number of occurrences
netflix_counts = netflix.groupby('type').size().reset_index(name='counts')
amazon_prime_counts = amazon_prime.groupby('type').size().reset_index(name='counts')
hulu_counts = hulu.groupby('type').size().reset_index(name='counts')
disney_plus_counts = disney_plus.groupby('type').size().reset_index(name='counts')

# Plot the results using a bar chart
ax.bar(netflix_counts['type'], netflix_counts['counts'], label='Netflix')
ax.bar(amazon_prime_counts['type'], amazon_prime_counts['counts'], label='Amazon Prime')
ax.bar(hulu_counts['type'], hulu_counts['counts'], label='Hulu')
ax.bar(disney_plus_counts['type'], disney_plus_counts['counts'], label='Disney+')

# Add a title and legend to the chart
ax.set_title('Number of Movies, TV Shows, and Other Types of Content on Each Streaming Platform')


# Read in the dataset
df = pd.read_csv('netflix_titles.csv')


# Select only the rows where the director is not null
df = df[df['director'].notnull()]

# Split the director column into a list of directors
directors = df['director'].str.split(',')

# Flatten the list of directors
directors = [director for sublist in directors for director in sublist]

# Count the number of times each director appears in the list
director_counts = {}
for director in directors:
    if director in director_counts:
        director_counts[director] += 1
    else:
        director_counts[director] = 1

# Sort the directors by their count, in descending order
sorted_directors = sorted(director_counts, key=director_counts.get, reverse=True)

# Select the top 25 directors
top_directors = sorted_directors[:25]

# Create a word cloud with the top 25 directors
wordcloud = WordCloud(background_color='white', width=800, height=400).generate(' '.join(top_directors))
fig10=plt.figure(figsize=(15,8))
# Display the word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")




z = dff.groupby(['rating']).size().reset_index(name='counts')
pieChart = px.pie(z, values='counts', names='rating', 
                  title='Distribution of Content Ratings on Netflix',
                  color_discrete_sequence=px.colors.qualitative.Set3)
                  
                  
#st.pyplot(pieChart)



# st.set_option('deprecation.showfileUploaderEncoding', False)

#openai.api_key = "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"  # Set this to your API key


st.write("WELCOME Ask A Question and have your Data analysis and Visualization Ready")

st.header("ROSALINE, AI/ML DATA ANALYTICS")

choices = ['Streaming Platform', 'Natural & Environment', 'Iphone', 'India Premier League']

# Create a selectbox widget
selected_choice = st.selectbox("Choose an Industry or Field:", choices)

st.markdown("We use our trained Dataset, If you want to use your own Dataset GO TO The PAGE 2")

#Rosaline = st.text_area("Question:")
button = st.button("Rosaline:")



    
if button: #and Rosaline:
    st.spinner("Rosaline is thinking...")
      #reply = generate_reply(Rosaline)
    #st.write(reply)
    if selected_choice == 'Streaming Platform':
      st.plotly_chart(fig1)
      st.write('.........')
      st.plotly_chart(pieChart)
      st.write('.........')
      st.plotly_chart(fig2)
      st.write('.........')
      st.plotly_chart(fig3)
      st.write('.........')
      st.plotly_chart(fig4)
      st.write('.........')
      st.plotly_chart(fig)
      st.write('.........')
      st.plotly_chart(fig9)
      st.write('.........')
      st.pyplot(fig10)
    elif selected_choice == 'Natural & Environment':
      st.plotly_chart(figure)
      st.write('.........')
      st.plotly_chart(figure1)
      st.write('.........')
      st.plotly_chart(figure2)
      st.write('.........')
      st.plotly_chart(figure3)
      st.write('.........')
      st.plotly_chart(figure4)
      st.write('.........')
      st.plotly_chart(figure5)

    elif selected_choice == 'Iphone':
#      st.write((highest_rated['Product Name']))
      st.plotly_chart(apo)
      st.write('.........')
      st.plotly_chart(apo1)
      st.write('.........')
      st.plotly_chart(apo3)
      st.write('.........')
      st.plotly_chart(apo4)
    elif selected_choice == 'India Premier League':
      st.plotly_chart(figo)
      st.write('.........')
      st.plotly_chart(figo1)
      st.write('.........')
      st.plotly_chart(figo2)
      st.write('.........')
      st.plotly_chart(figo3)
      st.plotly_chart(figo4)
      st.write('.........')
      st.plotly_chart(figo5)
      st.write('.........')
      st.plotly_chart(figo6)
      st.write('.........')
      st.plotly_chart(figo7)
      




