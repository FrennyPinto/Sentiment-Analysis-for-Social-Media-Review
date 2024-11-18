import pandas as pd
import requests   # Importing requests to extract content from a url
import re 
import nltk #library for natural language processing
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping. Used to scrap specific content 
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

# creating empty reviews list 
imdb_reviews=[]


for i in range(1,3): #seleact all the reviews from page 1 to 2
  ip=[]  
  url="https://www.imdb.com/title/tt0468569/reviews?ref_=tt_urv"
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  #soup = bs(response.content, "xml.parser")
  reviews = soup.find_all("div",attrs={"class","text"}) #can also give test show-more__control tag.....
  # Extracting the content under specific tags  #we have to cheak source code in inspect for extrectiong data from site
  for i in range(len(reviews)): #for every word in the review
    ip.append(reviews[i].text)  
 
  imdb_reviews=imdb_reviews+ip  # adding the reviews of one page to empty list which in future contains all the reviews


# writng reviews in a text file 
with open("imdb.txt","w",encoding='utf8') as output:
    output.write(str(imdb_reviews))
	

# Joinining/combining all the reviews into single paragraph 
ip_rev_string = " ".join(imdb_reviews) 


# Removing unwanted symbols
ip_rev_string = re.sub("[^A-Za-z" "]+"," ", ip_rev_string).lower() #convert the test to lowercase
ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string) #remove the numbers and replace them with the space.

# words that contained in movie reviews
ip_reviews_words = ip_rev_string.split(" ") #split the para into words. This show how many times each word has appeared. 


#TFIDF (Term Frequency Inverse Document Frequency)
#method used to convert the unstructured to structured data
# each review is considered as document
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(use_idf=True,ngram_range=(1, 3)) #uni, bi and tri
X = vectorizer.fit_transform(ip_reviews_words)

#read all the stopwords from file
with open(r"stop.txt") as sw:  #you can get the file on internet
    stop_words = sw.read()
   
stop_words = stop_words.split("\n") #every stop word will be indexed

stop_words.extend(["movie",'film',"Christian","Bale", "bale","joker","batman","Heath","heath", 'dark', 'knight', "ledger","nolan", "Nolan","Jonathan","story","time","vfx","character","director"]) 
#add these into stopword category.
#Removed names for extracting better review

ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]

# Joinining all the reviews into single paragraph 
# only the commom words from both will be kept, rest all will be removed
ip_rev_string = " ".join(ip_reviews_words)


#unigram wordcloud
wordcloud_ip = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400,
                      collocations=False
                     ).generate(ip_rev_string)
wordcloud_ip.to_file("Unigram.png")
plt.imshow(wordcloud_ip)


# positive words 
# Choose the path for +ve words stored in system
with open(r"positive-words.txt") as pos:
  poswords = pos.read().split("\n")


#### Positive word cloud ####
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400,
                      collocations=False
                     ).generate(ip_pos_in_pos)
wordcloud_pos_in_pos.to_file("PositiveWC.png")
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos)



#### negative word cloud #### 
#Choose path for -ve words stored in system
with open(r"negative-words.txt") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400,
                      collocations=False
                     ).generate(ip_neg_in_neg)
wordcloud_neg_in_neg.to_file("NagativeWC.png")
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)



WNL = nltk.WordNetLemmatizer() #this will perform stemming e.g dogs to dog

# Lowercase and tokenize
text = ip_rev_string.lower()

# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")

tokens = nltk.word_tokenize(text) #break the para into single word
text1 = nltk.Text(tokens) #This creates an object from the list of tokens.


# Remove extra chars and remove stop words.
text_content1 = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in tokens]



# Create a set of stopwords
stopwords_wc = set(STOPWORDS)
customised_words = ['price', 'great', 'Jonathan','nolan', 'joker', 'Batman'] # If you want to remove any particular word form text which does not contribute much in meaning


new_stopwords = stopwords_wc.union(customised_words) 


# Remove stop words
text_content = [word for word in text_content if word not in new_stopwords]
#text_content = [word for word in text_content if word not in stop_words]


# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]

#nltk.download('wordnet')
#nltk.download('omw-1.4')

# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]

#nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content)) #given bigram command for data to take two words
print(bigrams_list)

dictionary2 = [' '.join(tup) for tup in bigrams_list] #join bigrams without '' and ,
print (dictionary2)


# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(2,2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vocabulary = vectorizer.vocabulary_ #this will have the bigram and its index
#index corresponds to the position of the bigram in the columns of bag_of_words.
#The indices are assigned based on the frequency of the bigrams in the corpus. Higher frequency bigrams typically get lower indices.
feature_names = vectorizer.get_feature_names_out()


First_index = min(vectorizer.vocabulary_.values())
last_index = max(vectorizer.vocabulary_.values())

# Optionally, find the corresponding bigram (n-gram) with the last index
bigram_with_last_index = [bigram for bigram, index in vectorizer.vocabulary_.items() if index == last_index]

print("The last index in the vocabulary is:", last_index)
print("The bigram with the last index is:", bigram_with_last_index)


sum_words = bag_of_words.sum(axis=0) #sum of each bigram in the document
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()] 
#this return the frequency of each bigram or sum of the bigram count
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True) #sort in the reverse order of frequency.
print(words_freq[:100]) #show only first 100


# Generating wordcloud Bi gram
words_dict = dict(words_freq)
wordCloud = WordCloud(max_words= 200, height= 1000, width= 1500, stopwords=new_stopwords)
wordCloud.generate_from_frequencies(words_dict)

plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
wordCloud.to_file("Bigram.png")
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()


def perform_sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score

# Perform sentiment analysis
sentiment_score = perform_sentiment_analysis(ip_reviews_words[4])
print("Word for Sentiment Analysis: ", ip_reviews_words[4])
print("Sentiment Score:", sentiment_score)

#sentiment Analysis on Uni_Gram
sentiment_results_UniG = [(word, perform_sentiment_analysis(word)) for word in ip_reviews_words]
sentiment_df_UniG = pd.DataFrame(sentiment_results_UniG, columns=[" Uni-Gram", "Sentiment Score"]) # Convert the list of tuples into a pandas DataFrame
print(sentiment_df_UniG)


# Perform sentiment analysis on all bigram
sentiment_results_BiG = [(word, perform_sentiment_analysis(word)) for word in dictionary2]
sentiment_df_BiG = pd.DataFrame(sentiment_results_BiG, columns=[" Bi-Gram", "Sentiment Score"]) # Convert the list of tuples into a pandas DataFrame
print(sentiment_df_BiG)

