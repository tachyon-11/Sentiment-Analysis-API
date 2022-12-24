#Load libraries for deployment
from fastapi import FastAPI
from pydantic import BaseModel

#Load the libraries for our model
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#importing the training data
airplane_data=pd.read_csv('airline_sentiment_analysis.csv')

class sentiment_analyzer:
  def __init__( self, iterations ) :        
    self.lr = LogisticRegression(penalty='l2',max_iter=iterations,C=1,random_state=42)
    self.scr = 0
    #Count vectorizer for bag of words
    self.cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))

  def train(self, airplane_data):
    #Tokenization of text
    tokenizer=ToktokTokenizer()
    #Setting English stopwords
    stopword_list=nltk.corpus.stopwords.words('english')

    #Removing the html strips
    def strip_html(text):
      soup = BeautifulSoup(text, "html.parser")
      return soup.get_text()

    #Removing the square brackets
    def remove_between_square_brackets(text):
      return re.sub('\[[^]]*\]', '', text)

    #Removing the noisy text
    def denoise_text(text):
      text = re.sub(r'@[A-Za-z0-9]+','',text)
      text = strip_html(text)
      text = remove_between_square_brackets(text)
      return text
    
    #Apply function on text column
    airplane_data['text']=airplane_data['text'].apply(denoise_text)

    #Define function for removing special characters
    def remove_special_characters(text, remove_digits=True):
      pattern=r'[^a-zA-z0-9\s]'
      text=re.sub(pattern,'',text)
      return text
    
    #Apply function on text column
    airplane_data['text']=airplane_data['text'].apply(remove_special_characters)

    #set stopwords to english
    stop=set(stopwords.words('english'))

    #removing the stopwords
    def remove_stopwords(text, is_lower_case=False):
      tokens = tokenizer.tokenize(text)
      tokens = [token.strip() for token in tokens]
      if is_lower_case:
          filtered_tokens = [token for token in tokens if token not in stopword_list]
      else:
          filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
      filtered_text = ' '.join(filtered_tokens)    
      return filtered_text

    #Apply function on text column
    airplane_data['text']=airplane_data['text'].apply(remove_stopwords)

    #Stemming the text
    def simple_stemmer(text):
      ps=nltk.porter.PorterStemmer()
      text= ' '.join([ps.stem(word) for word in text.split()])
      return text
    
    #Apply function on text column
    airplane_data['text']=airplane_data['text'].apply(simple_stemmer)

    train, test = train_test_split(airplane_data, test_size=0.1)
    #normalized train reviews
    norm_train_reviews=train.text
    #Normalized test texts
    norm_test_reviews=test.text
    #transformed train reviews
    cv_train_reviews=self.cv.fit_transform(norm_train_reviews)
    #transformed test reviews
    cv_test_reviews=self.cv.transform(norm_test_reviews)

    #labeling the sentient data
    lb=LabelBinarizer()
    #transformed sentiment data
    train_sentiments= lb.fit_transform(train.airline_sentiment)
    test_sentiments = lb.fit_transform(test.airline_sentiment)

    X_train = cv_train_reviews
    Y_train = train_sentiments
    model.fit(X_train, Y_train)
    self.scr = accuracy_score(test_sentiments,self.predict(cv_test_reviews))

  # Function for model training    
  def fit( self, X, Y ) :  
    self.lr.fit(X,Y)
      
  def predict( self, X ) :    
    Y_pred = self.lr.predict(X)
    return Y_pred

  def answer( self, X):
    y_pred = self.lr.predict(self.cv.transform(X))
    if y_pred == 0:
      return 'negative'
    else:
      return 'positive'
  
  
model = sentiment_analyzer(iterations = 1000 )
model.train(airplane_data)


print(model.scr)

#For deploying API
app = FastAPI()

class request_body(BaseModel):
    review : str

@app.post('/predict')
def predict(data : request_body):
    test_data = [data.review]
    ans_sentiment = model.answer(test_data)
    
    return { 'sentiment' : ans_sentiment}
