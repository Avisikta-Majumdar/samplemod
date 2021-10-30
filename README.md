
## Python Package Development
As a data scientist, I am required to develop a package called **twtr** for twitter data processing and make it ready for the model development. 

#### What I have done to clean the tweets
1.   Imported libraries like pandas , nltk
2.   Read the csv file
3.   From the 15 columns take only 2 columns (airline_sentiment , text) and make another dataframe
4.   Removed twitter handles(@user)
5.   Converted all the text into lowercase
6.   Removed all the words which is having length is less than 3
7.   Converted **airline_sentimen**t column to numerical where positive got highest weightage and negative got lowest weightage
8.   Then on **Cleaned_text** column splited all the paragraphs & made a list
9.   used PorterStemmer for **Stemming**
10.   Removed **text**(which was having raw data)
**Now in twtr(dataframe) we have clean data **

## Thank you for visiting my GitHub profile ,let's connect on LinkedIn 
[LinkedIn](https://www.linkedin.com/in/avisikta-majumdar/)
