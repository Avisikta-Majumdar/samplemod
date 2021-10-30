#Importing libraries
import numpy as np
import re
import pandas as pd
import nltk


def data_preprocessing():
  from google.colab import drive
  drive.mount('/content/drive')

  path = '/content/drive/MyDrive/AlmaBetter/Modules/ Module 6 : ML & Data Engineering/6.2 - Python Modules and Project Setup/Twitter/data/raw/Tweets.csv'
  twtr = pd.read_csv(path)

  twtr.shape

  twtr.isnull().sum()

  twtr.head()

  twtr = twtr[['airline_sentiment' , 'text']]
  twtr.head(2)

  """## **Clean text**

  #### 1.   Basic Cleaning like URL
  """

  # string_with_nonASCII = "àa string withé fuünny charactersß."
  remove_URL =[]
  for single_text in twtr.text:
    #print(type(row) , '   ',row)
    single_text = re.sub(r'http\S+', '', single_text)
    remove_URL.append(single_text)

  for text in remove_URL:
    print(text)

  """Here I have made another column **Cleaned_text** """

  twtr['Cleaned_text'] = remove_URL
  twtr.head()

  """#### 2.    Twitter Handle"""

  def remove_pattern(input_txt, pattern):
      r = re.findall(pattern, input_txt)
      for i in r:
          input_txt = re.sub(i, '', input_txt)
          
      return input_txt

  # remove twitter handles (@user)
  twtr['Cleaned_text'] = np.vectorize(remove_pattern)(twtr.Cleaned_text, "@[\w]*")
  twtr.head()

  #Removed the short words like i , is ,ok,hi hey 
  twtr['Cleaned_text'] = twtr['Cleaned_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

  """#### 3.   Non-ASCII"""

  # string_with_nonASCII = "àa string withé fuünny charactersß."
  remove_noASCII =[]
  for row in twtr.Cleaned_text:
    #print(type(row) , '   ',row)
    row = row.encode("ascii", "ignore")
    remove_noASCII.append(row)

  for text in remove_noASCII:
    print(text)

  twtr.head()

  """## 2. Standardize text(Lowercasing , Stemming , Lemmatization , Stopword Removal , Normalization)

  [Read this ](https://www.kdnuggets.com/2019/04/text-preprocessing-nlp-machine-learning.html)
  *   **Lowercasing**


  """

  #Lowercasing is a type of text preprocessing techniques
  twtr.Cleaned_text = twtr.Cleaned_text.apply(lambda x: x.lower())



  twtr.head()

  tokenized_tweet = twtr.Cleaned_text.apply(lambda x: x.split())
  tokenized_tweet.head()

  """

  *   **Stemming**

  """

  stemmer = nltk.stem.PorterStemmer()
  tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
  tokenized_tweet.head()

  twtr.Cleaned_text = tokenized_tweet

  twtr.head()

  twtr.drop(['text'] ,axis=1, inplace=True)

  print(twtr.airline_sentiment.value_counts())
  #Converting the airline_sentiment column from categorical to numerical
  twtr.airline_sentiment = twtr.airline_sentiment.replace({'negative':-1 , 'neutral':0 , 'positive':1 })

  twtr.head()

  #Already done , no need to do it again
  # twtr.to_csv('/content/drive/MyDrive/AlmaBetter/Modules/ Module 6 : ML & Data Engineering/6.2 - Python Modules and Project Setup/Twitter/samplemod/twtr/Clean_Tweets.csv' , index = False)

