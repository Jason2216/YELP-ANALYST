#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.stem import SnowballStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix


# In[2]:


df=pd.read_csv('/kaggle/input/yelp-reviews-dataset/yelp.csv')


# In[3]:


df.head(5)


# 

# In[4]:


df.shape


# 

# In[5]:


df.columns


# In[6]:


df.info()


# 

# In[7]:


df.isnull().sum()


# In[8]:


df=df.drop(['business_id','date','review_id','type','user_id','cool','useful','funny'],axis=1)


# In[9]:


df.head(5)


# ##### data analyst and visulization

# In[10]:


#lets make some text analysis and calculate the the length of each review
list_review=[]
for i in df['text']:
    list_review.append(len(i))


# ###### Creating a new columns text_length for reviews

# In[11]:


df['text_length']=list_review


# In[12]:


df.head(5)


# In[13]:


plt.figure(figsize=(12,5))
df['stars'].value_counts().plot(kind='bar',alpha=0.5,color='green',label='ratings')
plt.legend()
plt.show()


# # import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(12,5))
# df[df['stars']==1]['text_length'].plot(bins=35,alpha=0.5,kind='hist',color='red',label='rating 1')
# plt.legend()
# plt.xlabel('text_length')
# plt.show()

# In[15]:


plt.figure(figsize=(12,5))
df[df['stars']==2]['text_length'].plot(bins=35,kind='hist',color='yellow',alpha=0.5,label='2 ratings')
plt.xlabel('text length')
plt.legend()
plt.show()


# In[16]:


plt.figure(figsize=(12,5))
df[df['stars']==3]['text_length'].plot(bins=35,kind='hist',color='brown',alpha=0.5,label='3 ratings')
plt.xlabel('text length')
plt.legend()
plt.show()


# In[17]:


plt.figure(figsize=(12,5))
df[df['stars']==4]['text_length'].plot(bins=35,kind='hist',color='orange',alpha=0.5,label='4 ratings')
plt.xlabel('text length')
plt.legend()
plt.show()


# In[18]:


plt.figure(figsize=(12,5))
df[df['stars']==5]['text_length'].plot(bins=35,kind='hist',color='pink',alpha=0.5,label='5 ratings')
plt.xlabel('text length')
plt.legend()
plt.show()


# In[19]:


df['stars']=np.where(df['stars']>3,1,0)


# ##### Here is the changed senario

# In[20]:


sns.countplot(df['stars'])


# ### created two list for extracting the positive(1-class) and negative reviews(0-class) 

# In[21]:


positive=[]
negative=[]
positive_reviews=df[df['stars']==1]['text']
negative_reviews=df[df['stars']==0]['text']


# #### The extract_postive function will do following things:
# #### 1-It will tokenize the text from positive reviews
# #### 2-Remove the stopwords
# #### 3-Convert into lowercase

# In[22]:


def extract_positive(positive_reviews):
    global positive
    words = [word.lower() for word in word_tokenize(positive_reviews) if word.lower() not in stopwords.words("english") and word.lower().isalpha()]
    positive=positive+words


# #### Similar functionality but for negative reviews

# In[23]:


def extract_negative(negative_reviews):
    global negative
    words = [word.lower() for word in word_tokenize(negative_reviews) if word.lower() not in stopwords.words("english") and word.lower().isalpha()]
    negative=negative+words


# ### WordCloud

# In[25]:


from wordcloud import WordCloud
pos_review_cloud=WordCloud(width=600,height=400).generate(" ".join(positive_reviews))
plt.figure(figsize=(10,8),facecolor='k')
plt.imshow(pos_review_cloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# #### As we can clearly observe that there are postive words like good,great,amazing etc

# #### For Negative Reviews

# In[26]:


neg_review_cloud=WordCloud(width=600,height=400,background_color='white').generate(" ".join(negative_reviews))
plt.figure(figsize=(10,8),facecolor='k')
plt.imshow(neg_review_cloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# #### Some mixed words like better,good ,try could be a part of some suggestion or negative review

# # Section:3 Modelling

# In[27]:


stemmer = SnowballStemmer("english")

def ReadyText(message):
    
    message = message.translate(str.maketrans('', '', string.punctuation))
    words = [stemmer.stem(word) for word in message.split() if word.lower() not in stopwords.words("english")]
    
    return " ".join(words)

df["text"] = df["text"].apply(ReadyText)
df.head(n = 10)    


# In[28]:


y=df['stars']
x=df['text']


# ### In this section we have spliited our dataset into training set and test test with a ratio of 0.2.The model will be trained  on training set and will be evaluated on the unseen data i.e test set

# In[29]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)


# In[30]:


print("x_train shape :",x_train.shape)
print("x_test shape  :",x_test.shape)
print("y_train shape :",y_train.shape)
print("y_tets shape  :",y_test.shape)


# ### As we know that nachine learning algorithms don't accept as text as input so we have to convert into vectors .For this  we have used CountVectorizer for converting the text into vectors

# ### Here we will build models using three classification algorithms 
# 

# #### 1:Logistic Regression

# In[31]:


cv=CountVectorizer()
lr=LogisticRegression(max_iter=10000)
x_train=cv.fit_transform(x_train)
lr.fit(x_train,y_train)
pred_1=lr.predict(cv.transform(x_test))
score_1=accuracy_score(y_test,pred_1)


# ### Accuracy Score :Logistic Regression

# In[32]:


print(score_1*100)


# ### Making Predictions :Logistic Regression

# In[33]:


print(lr.predict(cv.transform(['Thats a very good dish,I like it'])))


# #### It predicted as a positive review.So our model is working Fine

# ### Confusion Matrix :Logistic Regression

# In[34]:


cm = confusion_matrix(y_test, pred_1)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Logistic Regression Model - Confusion Matrix")
plt.xticks(range(2), ["Positve Review","Negative Review"], fontsize=16)
plt.yticks(range(2), ["Positive Review","Negative Review"], fontsize=16)
plt.show()


# ### Navie Bayes:

# In[35]:


nb=MultinomialNB()
nb.fit(x_train,y_train)
pred_2=nb.predict(cv.transform(x_test))
score_2=accuracy_score(y_test,pred_2)


# In[36]:


print(score_2*100)


# In[37]:


print(nb.predict(cv.transform(['the dish is just ok'])))


# ### Confusion Matrix:Naive Bayes

# In[38]:


cm = confusion_matrix(y_test, pred_2)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Naive Bayes - Confusion Matrix")
plt.xticks(range(2), ["Positve Review","Negative Review"], fontsize=16)
plt.yticks(range(2), ["Positive Review","Negative Review"], fontsize=16)
plt.show()


# ### Support Vetor Machines:

# In[39]:


svm=SVC()
svm.fit(x_train,y_train)
pred_3=svm.predict(cv.transform(x_test))
score_3=accuracy_score(y_test,pred_3)


# ### Accuracy Score:Support Vector Machine

# In[40]:


print(score_3*100)


# ### Predictions:Support Vector Machine

# In[41]:


print(svm.predict(cv.transform(['I love this place'])))


# ### Predicted Correctly again!

# ### Confusin Matrix:Support Vector Machine

# In[42]:


cm = confusion_matrix(y_test, pred_3)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Support Vector Machine - Confusion Matrix")
plt.xticks(range(2), ["Positve Review","Negative Review"], fontsize=16)
plt.yticks(range(2), ["Positive Review","Negative Review"], fontsize=16)
plt.show()


# 
