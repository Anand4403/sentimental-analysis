Intel i5 reviews - Sentiment Analysis
Analyzing the intel i5 dataset and building classification models to predict if the sentiment of a given input sentence is positive or negative
### Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pickle
import re

Exploratory Data Analysis
#Load the data

data = pd.read_csv(r"intel_i5.tsv", delimiter = '\t', quoting = 3)

print(f"Dataset shape : {data.shape}")
Dataset shape : (20, 4)

data.head()
URL	review_rating	review_title	review_body
0	https://www.amazon.in/Intel-i5-12400-Desktop-P...	5	Working fine with ASUS prime Z690-P WiFi D4 mo...	Box contents: Intel core i5 12400 processor ch...
1	https://www.amazon.in/Intel-i5-12400-Desktop-P...	5	Super 👌 I am already used	I am build my first pc this processor
2	NaN	5	I5	Good and efficient 👍
3	NaN	5	Good	I love this
4	NaN	4	Good CPU in this range	Getting optimal speed without any overheating ...


#Column names

print(f"Feature names : {data.columns.values}")
Feature names : ['URL' 'review_rating' 'review_title' 'review_body']

#Creating a new column 'length' that will contain the length of the string in 'verified_reviews' column

data['length'] = data['review_body'].apply(len)
data.head()
	URL	review_rating	review_title	review_body	length
0	https://www.amazon.in/Intel-i5-12400-Desktop-P...	5	Working fine with ASUS prime Z690-P WiFi D4 mo...	Box contents: Intel core i5 12400 processor ch...	202
1	https://www.amazon.in/Intel-i5-12400-Desktop-P...	5	Super 👌 I am already used	I am build my first pc this processor	37
2	NaN	5	I5	Good and efficient 👍	20
3	NaN	5	Good	I love this	11
4	NaN	4	Good CPU in this range	Getting optimal speed without any overheating ...	51

#Randomly checking for 10th record

print(f"'verified_reviews' column value: {data.iloc[10]['review_body']}") #Original value
print(f"Length of review : {len(data.iloc[10]['review_body'])}") #Length of review using len()
print(f"'length' column value : {data.iloc[10]['length']}") #Value of the column 'length'

'verified_reviews' column value: Nice
Length of review : 4
'length' column value : 4
We can see that the length of review is the same as the value in the length column for that record

Datatypes of the features![download](https://github.com/user-attachments/assets/b682d6af-e1ea-4186-b7a5-9cabc2934bda)



Analyzing 'rating' column
This column refers to the rating of the variation given by the user
len(data)
20
#Distinct values of 'rating' and its count  

print(f"Rating value count: \n{data['review_rating'].value_counts()}")
Rating value count: 
review_rating
5    18
4     1
1     1
Name: count, dtype: int64

#Bar plot to visualize the total counts of each rating

data['rating'].value_counts().plot.bar(color = 'red')
plt.title('Rating distribution count')
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()
![download](https://github.com/user-attachments/assets/e72582c5-9994-46a0-bd71-bbafd5e421ca)

#Finding the percentage distribution of each rating - we'll divide the number of records for each rating by total number of records

print(f"Rating value count - percentage distribution: \n{round(data['review_rating'].value_counts()/data.shape[0]*100,2)}")

Rating value count - percentage distribution: 
review_rating
5    90.0
4     5.0
1     5.0
Name: count, dtype: float64

Let's plot the above values in a pie chart

fig = plt.figure(figsize=(7,7))

colors = ('red', 'green', 'blue')

wp = {'linewidth':1, "edgecolor":'black'}

tags = data['review_rating'].value_counts()/data.shape[0]

explode=(0.1,0.1,0.1,)

tags.plot(kind='pie', autopct="%1.1f%%", shadow=True, colors=colors, startangle=90, wedgeprops=wp, explode=explode, label='Percentage wise distrubution of rating')

from io import  BytesIO

graph = BytesIO()

fig.savefig(graph, format="png")
![download](https://github.com/user-attachments/assets/0bb5f61c-a6b8-4943-8c86-2bcb7ca9c17f)

### Analyzing 'review_rating' column

This column refers to the feedback of the verified review
#Distinct values of 'feedback' and its count 

print(f"Feedback value count: \n{data['review_rating'].value_counts()}")
Feedback value count: 
review_rating
5    18
4     1
1     1
Name: count, dtype: int64

#Bar graph to visualize the total counts of each feedback

data['review_rating'].value_counts().plot.bar(color = 'blue')
plt.title('Feedback distribution count')
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()

![download](https://github.com/user-attachments/assets/b897492f-d567-4b68-9008-dbc14514c766)




#Finding the percentage distribution of each rating - we'll divide the number of records for each feedback by total number of records

print(f"Feedback value count - percentage distribution: \n{round(data['review_rating'].value_counts()/data.shape[0]*100,2)}")
Feedback value count - percentage distribution: 
review_rating
5    90.0
4     5.0
1     5.0
Name: count, dtype: float64

Feedback distribution

91.87% reviews are positive
8.13% reviews are negative

fig = plt.figure(figsize=(7,7))

colors = ('blue', 'green', 'yellow')

wp = {'linewidth':1, "edgecolor":'black'}


tags = data['review_rating'].value_counts()/data.shape[0]

explode=(0.1,0.1,0.1)

tags.plot(kind='pie', autopct="%1.1f%%", shadow=True, colors=colors, startangle=90, wedgeprops=wp, explode=explode, label='Percentage wise distrubution of rating')


![download](https://github.com/user-attachments/assets/23f43a5a-c229-4f3a-a471-bda34fbcefbf)



[SENTIMENTAL ANALYSIS OF REVIEWS (1).pptx](https://github.com/user-attachments/files/16238716/SENTIMENTAL.ANALYSIS.OF.REVIEWS.1.pptx)

