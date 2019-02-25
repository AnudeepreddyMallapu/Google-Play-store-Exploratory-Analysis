#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import re
from IPython.display import display
pd.options.display.max_rows=None
from scipy import stats
from statsmodels.formula.api import ols 
from IPython.display import display, Markdown
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


playstore=pd.read_csv(r"C:\Users\andee\googleplaystore.csv")


# In[8]:


#Checking out the info, datatype, number of rows, columns
playstore.info()
playstore.head()


# In[9]:


playstore.drop(index=10472,inplace=True)


# In[72]:


def change_size(size):
    kb= size.str.endswith("k")
    MB=  size.str.endswith("M")
    other= ~(kb|MB)
    size.loc[kb]= size[kb].str.replace("k","").astype("float")/1024
    size.loc[MB]= size[MB].str.replace("M","").astype("float")
    size.loc[other] = float(0.0)
change_size(playstore.Size)
playstore.columns= [x.replace(" ","_") for x in  playstore.columns]
playstore.Installs= np.log(playstore.Installs.str.replace("[+,]","").astype("int64")+1)
playstore.Reviews= np.log(playstore.Reviews.astype("int")+1)
playstore.Price= playstore.Price.str.replace("[$,]","").astype("float")
playstore.Size=playstore.Size.astype("float")
#playstore.Type= pd.get_dummies(playstore.Type,drop_first=True)


# In[74]:


playstore.info()


# In[12]:


#Dealing with Null Values


# In[19]:


total = playstore.isnull().sum().sort_values(ascending=False)
percent = (playstore.isnull().sum()/playstore.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
display(missing_data.head(6))
print("Before Cleaning")
display(playstore.shape)


# In[20]:


playstore.Rating.fillna(method='ffill',inplace=True)
playstore.dropna(how ='any', inplace = True)
print("After Cleaning")
playstore.shape


# In[21]:


sns.pairplot(playstore,kind="scatter",hue='Type')


# 1- Most of the rating are on or above 4.
# 
# 2- Rating and reiveiws have a positive correlation
# 
# 3- Free apps reviews are more spread then paid apps revies.
# 
# 4- Reviews and Installs have a strong correlation.
# 
# 5- Instalation is almost normaly destributed for both paid and free app
# 
# 6- most of the app size is below 50 MB
# 

# In[22]:


playstore.Category.value_counts().plot(kind='bar',figsize=(18,6))


# By category destribution we can see that most of the app are from top 3 category.

# In[24]:


playstore.groupby("Type")["Type"].value_counts().plot(kind='pie',autopct='%1.1f%%' );


# Only about 7% app are paid App.

# In[25]:


sns.catplot(x="Type",y="Installs",kind='box',data=playstore,  height=4, aspect=2/1);


# Here I found that Free apps are way more downloaded than paid apps.

# In[26]:


sns.catplot(x="Type",y="Rating",kind='box',data=playstore,  height=4, aspect=2/1);


# Here Paid apps have better rating than free apps.

# Q.Are paid have significantly better rating than free apps?

# In[27]:


#null Hypothisis- avg rating are same for paid and free app
model_name = ols('Rating ~ C(Type)', data=playstore).fit()
model_name.summary()


# Here Paid app have significantly higher rating than free app.

# In[28]:


sns.catplot(x="Type",y="Size",kind='box',data=playstore, height=4, aspect=2/1);


# Size are almost same for the paid and free app.

# ### Price of the apps

# In[29]:


sns.catplot("Price","Category",data=playstore[playstore.Price>0],height=10,aspect=1.5,hue="Content_Rating",s=10,alpha=.3);


# Lets look into high price app
# 
# 

# In[30]:


playstore[playstore.Price>200][["Category","App","Price"]]


# They all look junk app, lets remove them and plot the graph again
# 
# 

# In[31]:


sns.catplot("Price","Category",data=playstore[playstore.Price>0],height=10,aspect=1.5,hue="Content_Rating",s=10,alpha=.3)


# Here we found
# 
# 1-Most of the app are priced well below $50.
# 
# 2-Most of the paid are from family, Medical and games category.

# ### Paid apps Content Rating
# 

# In[33]:


playstore[playstore.Price>0].groupby("Content_Rating")["App"].count().plot(kind='bar')


# Most of the paid are for everyone's use, With some of the app focused on teen.

# ## EDA for category and Genres
# ### Size of app

# In[34]:


sns.catplot("Size","Category",data=playstore,height=10,aspect=2/1,c=1/1000,s=10,alpha=0.2,hue="Content_Rating")


# Most of the large size app are form Family and Game category
# 

# ### Rating by Category
# 

# In[37]:


sns.catplot(x="Category",y="Rating",kind='box',data=playstore, height=8, aspect=2/1);
plt.xticks(rotation=90);


# Average rating accross the category is around 4.2. But are the same across the category.
# 
# 

# In[38]:


model_name = ols('Rating ~ C(Category)', data=playstore).fit()


# In[39]:


model_name.summary()


# By looking the F-statistic and P valuse we can say rating is significantly diferent accross the category.
# 
# 

# ### Installation by category

# In[41]:


sns.catplot(x="Category",y="Installs",kind='violin',data=playstore, height=8, aspect=2/1);
sns.lineplot(x=range(0,len(playstore.Category.unique())),y=playstore["Installs"].mean(),)
plt.xticks(rotation=90);


# By looking the above graph we can see that Education, Entertanments,Game ,Shopping,Social and weather are the most downlaod app. Where as Business , Events and medical are the least downloaded app.
# 

# In[43]:


playstore.Genres.value_counts().head(30).plot("bar",figsize=(18,6))


# By looking the above graph we found that Apps are well destributed among top 20 Genres
# 

# ### Relationship Between rating and reviews

# In[44]:


sns.regplot(playstore.Rating,playstore.Reviews,color="g",x_estimator=np.mean);


# We found that ratings and reviews have positive correlation

# ### App Reviews

# In[46]:


app_reviews= pd.read_csv(r"C:\Users\andee\googleplaystore_user_reviews.csv")


# In[47]:


app_reviews.head()


# In[48]:


app_reviews.info()


# In[49]:


#There a lot of null values across all the columns. Lets remove them

app_reviews.isnull().sum().sort_values()


# In[50]:


app_reviews.dropna(inplace=True)
app_reviews.isnull().sum().sort_values()


# In[51]:


app_reviews.Sentiment.value_counts()


# There are almost 3 time positive review than negative.

# ## Joining app data with app reviews¶
# 

# In[52]:


combined_data= playstore[["App","Type","Category","Genres","Content_Rating"]].merge(app_reviews,how="inner",left_on="App",right_on="App")


# In[54]:


combined_data.head()                                                                


# ## Free app Vs Paid app Reviews
# 

# ### Lets start with comparing sentimets for paid and free apps
# 

# In[56]:


temp_type=(combined_data.groupby(["Type","Sentiment"])["App"].count()/combined_data.groupby(
    ["Type"])["App"].count()).reset_index(level=[0,1])

greenBars= temp_type[temp_type.Sentiment=='Positive']["App"]
orangeBars = temp_type[temp_type.Sentiment=='Negative']["App"]
blueBars = temp_type[temp_type.Sentiment=='Neutral']["App"]
r= list(range(0,len(temp_type.Type.unique())))


# In[57]:


barWidth = 0.85
names = temp_type.Type.unique()


# In[60]:


# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth)
# Create blue Bars
plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white', width=barWidth)
# Custom x axis
plt.xticks(r, names,rotation=90)
plt.xlabel("group")
#plt.legend() 
# Show graphic
plt.show()


# We found that
# 1. Paid apps have high number of positive reviews 80% and very less neutral reviews.
# 
# 2. Free Apps have just about 60% of positive reviews with almost 10% of neutral reviews.
# 

# ### lets check category wise for paid and free app separatly
# 

# In[62]:


#Analysis of sentimats by category for paid apps
temp_cat_paid=(combined_data[combined_data.Type=="Paid"].groupby(["Category","Sentiment"])["App"].count()/combined_data[combined_data.Type=="Paid"].groupby(
    ["Category"])["App"].count()).reset_index(level=[0,1])

greenBars= temp_cat_paid[temp_cat_paid.Sentiment=='Positive']["App"]
orangeBars = temp_cat_paid[temp_cat_paid.Sentiment=='Negative']["App"]
blueBars = temp_cat_paid[temp_cat_paid.Sentiment=='Neutral']["App"]
plt.figure(figsize=(15,8))

r= list(range(0,len(temp_cat_paid.Category.unique())))

barWidth = 0.85
names = temp_cat_paid.Category.unique()
# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth)
# Create blue Bars
plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], 
        color='#a3acff', edgecolor='white', width=barWidth)
plt.axhline(y=.75,color='r', linestyle='-')
# Custom x axis
plt.xticks(r, names,rotation=90)
plt.xlabel("group")
#plt.legend() 
# Show graphic
plt.show()


# Findings:
# 
#     1- Most of the paid medial app have positive reviews.
#     
#     2- Most of the paid app category have about 80% positive reviews except Game.
#     
#     3- Game have below 60% positive reviews with almost 40% negative reviews.

# ### Analysis of sentimats by category for Free apps
# 
# 

# In[64]:


temp_cat_free=(combined_data[combined_data.Type=="Free"].groupby(["Category","Sentiment"])["App"].count()/combined_data[combined_data.Type=="Free"].groupby(
    ["Category"])["App"].count()).reset_index(level=[0,1])

greenBars= temp_cat_free[temp_cat_free.Sentiment=='Positive']["App"]
orangeBars = temp_cat_free[temp_cat_free.Sentiment=='Negative']["App"]
blueBars = temp_cat_free[temp_cat_free.Sentiment=='Neutral']["App"]

plt.figure(figsize=(15,8))

r= list(range(0,len(temp_cat_free.Category.unique())))

barWidth = 0.85
names = temp_cat_free.Category.unique()
# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth)
# Create blue Bars
plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], 
        color='#a3acff', edgecolor='white', width=barWidth)
plt.axhline(y=.75,color='r', linestyle='-')
# Custom x axis
plt.xticks(r, names,rotation=90)
plt.xlabel("group")
#plt.legend() 
# Show graphic
plt.show()


# Findings:
#     
#     1- Average positive reviews are about 63% as compare to 80% for paid app
#     
#     2- Free apps have high numbers of neutral reviews 
#     
#     3- Free apps are more distrubuted across category 
#     
#     4- Free comics are very popular, with having 90% positive reviews

# ## Sentimet Polarity

# ### Free vs Paid
# 

# In[67]:


sns.catplot("Type","Sentiment_Polarity",data= combined_data,alpha=.3);


# We found that:
# 
#     1- Most of the Paid apps have positive polarity
#     
#     2- Free apps have a lot of strong nagetive reviews

# ### Deep dive into free apps polarity
# 
# 

# In[68]:


sns.catplot("Category","Sentiment_Polarity",data= 
            combined_data[combined_data.Type=="Free"],alpha=.2,height=8,aspect=1.5);
plt.xticks(rotation=90);


# Based on the above graph:
# 
#     Category having strong Negative reviews are:
#     
#     1- Game
#     
#     2- Family
#     
#     3- Dating
#     
#     4- Entertainment
#     
#     5- Traivels and Local

# In[ ]:




