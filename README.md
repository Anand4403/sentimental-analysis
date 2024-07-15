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
