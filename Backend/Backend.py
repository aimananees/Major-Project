###Libraries used###
from flask import Flask,jsonify 	#Flask-main library for making the Backend, jsonify-library for returning payload as JSON.
import requests 					#Library for requesting a URL in order to extract data.
import csv    						#Library for handling CSV files.
import re     						#Regex handler library.

import tweepy     					#Library for accessing Twitter API.
from tweepy import OAuthHandler
import os
import pickle


import numpy as np
import pandas as pd
from bs4 import BeautifulSoup 
import re
from nltk.corpus import stopwords
import nltk.data
from sklearn.ensemble import RandomForestClassifier #Classifier used for classification
from gensim.models import Word2Vec 	#Library for accessing Word2Vec
import preprocessor as p 			#Library for preprocessing twitter tweets




###Consumer API keys for using the API's###
API_KEY="la0kUGyfuWZCik55R2O0PZQiD"
API_SECRET_KEY="R3EbVIZGqUsv4joIfBzrVzZfct2oaFDNkhXSm26pYgGt9kNPe3"

###Access tokens###
ACCESS_TOKEN="1061229681469472770-4oz350AurHB0QiVgzudhOLoEf56Ok7"
ACCESS_TOKEN_SECRET="bjnCO7hKIKeWa4KvM6qSM4xddKp7l0nGH6haHFG93CFuj"

auth = OAuthHandler(API_KEY,API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN,ACCESS_TOKEN_SECRET)
api = tweepy.API(auth,timeout=10)
app = Flask(__name__)				#Creates an instance of the Flask class for our Backend.

"""
Preprocessing the incoming tweets:
(1) Getting rid of links present in the beginning, middle or end of a tweet.
(2) Lower-casing the tweets.
(3) Converting Contractions(I'm -> I am).
(4) Removing non-words from the tweets.
(5) Getting rid of extra spaces.
"""
def preprocessing(tweet):
	stops = set(stopwords.words("english"))     
	tweet=p.clean(tweet)
	tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s"," ",tweet)
	tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s"," ",tweet)
	tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$"," ",tweet)
	tweet = tweet.lower()
	tweet = re.sub(r"that's","that is",tweet)
	tweet = re.sub(r"there's","there is",tweet)
	tweet = re.sub(r"what's","what is",tweet)
	tweet = re.sub(r"where's","where is",tweet)
	tweet = re.sub(r"it's","it is",tweet)
	tweet = re.sub(r"who's","who is",tweet)
	tweet = re.sub(r"i'm","i am",tweet)
	tweet = re.sub(r"she's","she is",tweet)
	tweet = re.sub(r"he's","he is",tweet)
	tweet = re.sub(r"they're","they are",tweet)
	tweet = re.sub(r"who're","who are",tweet)
	tweet = re.sub(r"ain't","am not",tweet)
	tweet = re.sub(r"wouldn't","would not",tweet)
	tweet = re.sub(r"shouldn't","should not",tweet)
	tweet = re.sub(r"can't","can not",tweet)
	tweet = re.sub(r"couldn't","could not",tweet)
	tweet = re.sub(r"won't","will not",tweet)
	tweet = re.sub(r"\W"," ",tweet)
	tweet=tweet.split()
	words = [w for w in tweet if not w in stops]
	tweet = " ".join(tweet)
	return tweet

#A function for accessing the training datasets
def accessing_train_datasets(file_name,file_type):
	desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 
	csv_file = pd.read_csv(desktop+"/Git/Major-Project/Backend/"+file_name+"."+file_type, header=0,\
					delimiter="\t", quoting=3)
	return csv_file

#A function for accessing the test datasets
def accessing_test_datasets(file_name,file_type):
	desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 
	csv_file = pd.read_csv(desktop+"/Git/Major-Project/Backend/"+file_name+"."+file_type)
	return csv_file


def review_train_wordlist(review, remove_stopwords=False):
	review_text = BeautifulSoup(review).get_text()
	review_text = re.sub("[^a-zA-Z]"," ",review)
	words = review.lower().split()
	if remove_stopwords:
		stops = set(stopwords.words("english"))     
		words = [w for w in words if not w in stops]
	
	return(words)

def review_test_wordlist(review, remove_stopwords=False):
	words = review.lower().split()	
	return(words)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_sentences(review, tokenizer, remove_stopwords=False):
	# 1. Using nltk tokenizer
	raw_sentences = tokenizer.tokenize(review.strip())
	sentences = []
	# 2. Loop for each sentence
	for raw_sentence in raw_sentences:
		if len(raw_sentence)>0:
			sentences.append(review_train_wordlist(raw_sentence,\
											remove_stopwords))

	# This returns the list of lists
	return sentences

def generating_sentences():
	sentences = []
	print("Parsing sentences from training set")
	for review in accessing_train_datasets('train','tsv')['review']:
		sentences += review_sentences(review, tokenizer)

	return sentences

def building_model():
	num_features = 300  # Word vector dimensionality
	min_word_count = 40 # Minimum word count
	num_workers = 4     # Number of parallel threads
	context = 10        # Context window size
	downsampling = 1e-3 # (0.001) Downsample setting for frequent words

	# Initializing the train model
	from gensim.models import word2vec
	print("Training model....")
	model = word2vec.Word2Vec(generating_sentences(),\
							  workers=num_workers,\
							  size=num_features,\
							  min_count=min_word_count,\
							  window=context,
							  sample=downsampling)

	# To make the model memory efficient
	model.init_sims(replace=True)
  	# Saving the model for later use. Can be loaded using Word2Vec.load()
	model_name = "Sentiment Analysis Model"
	model.save(model_name)

def featureVecMethod(words, model, num_features):
	# Pre-initialising empty numpy array for speed
	featureVec = np.zeros(num_features,dtype="float32")
	nwords = 0
	
	#Converting Index2Word which is a list to a set for better speed in the execution.
	index2word_set = set(model.wv.index2word)
	
	for word in  words:
		if word in index2word_set:
			nwords = nwords + 1

			featureVec = np.add(featureVec,model[word])

	
	# Dividing the result by number of words to get average
	featureVec = np.divide(featureVec, (nwords+0.0000000000001))
	return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
	counter = 0
	reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
	for review in reviews:
		# Printing a status message every 1000th review
		if counter%1000 == 0:
			print("Review %d of %d"%(counter,len(reviews)))
			
		reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
		counter = counter+1
		
	return reviewFeatureVecs

def calculate_average_feature_vector_train_data(model, num_features):
	clean_train_reviews = []
	for review in accessing_train_datasets('train','tsv')['review']:
		clean_train_reviews.append(review_train_wordlist(review, remove_stopwords=True))
	
	trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
	return trainDataVecs

def calculate_average_feature_vector_test_data(model, num_features,companyName):
	clean_test_reviews = []
	for review in accessing_test_datasets(companyName+"_tweets","csv")["Tweet"]:
		clean_test_reviews.append(review_test_wordlist(review,remove_stopwords=True))
		
	testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)
	return testDataVecs

def train_model(model,num_features):
	forest = RandomForestClassifier(n_estimators = 100)
	
	print("Fitting random forest to training data....")    
	forest = forest.fit(calculate_average_feature_vector_train_data(model,num_features), accessing_train_datasets('train','tsv')["sentiment"])
	return forest

def test_model(forest, testDataVecs,companyName):
	print(testDataVecs.shape)
	result = forest.predict(testDataVecs)
	output = pd.DataFrame(data={"text":accessing_test_datasets(companyName+'_tweets','csv')['Tweet'], "sentiment":result,"date":accessing_test_datasets(companyName+'_tweets','csv')['Date']})
	output.to_csv(companyName+"_output.csv")



"""
GET Reuqest URL: http://127.0.0.1:5000/majorProject/api/v1.0/fetchStockData/<quandlCode>/<companyCode>/<startDate>/<endDate>
Arguments:
quandlCode - a specific code for the stock
companyCode - NSE code for the stock
startDate, endDate - fetching stock data between a start date and end date

EXAMPLE: http://127.0.0.1:5000/majorProject/api/v1.0/fetchStockData/NSE/DABUR/01-01-2018/02-12-2018
Above URL fetches stock data for DABUR from 01-01-2018 - 02-12-2018
"""
@app.route('/majorProject/api/v1.0/fetchStockData/<quandlCode>/<companyCode>/<startDate>/<endDate>', methods=['GET'])
def fetchStockData(quandlCode, companyCode, startDate, endDate):

	URL="https://www.quandl.com/api/v3/datasets/"+quandlCode+"/"+companyCode+"?start_date="+startDate+"&end_date="+endDate+"&api_key=x7fbrZQv8kYCyb61KSiY"
	fetchStockData = requests.get(url=URL)	#Fetching data from Quandl Repository using the above URL
	fetchStockData=fetchStockData.json()

	###Storing the data in a csv format###
	with open(fetchStockData['dataset']['name']+'.csv',"w") as file:
		csvWriter = csv.writer(file,delimiter=',')
		csvWriter.writerow(['Date','Open','High','Low','Last','Close','Total Trade Quantity','Turnover (Lacs)'])
		csvWriter.writerows(fetchStockData['dataset']['data'])

	###Returning the payload in JSON format as a reponse to the GET Request###
	return jsonify({'name':fetchStockData['dataset']['name']},{'column_names':fetchStockData['dataset']['column_names']},{'data':fetchStockData['dataset']['data']})


"""
GET Reuqest URL: http://127.0.0.1:5000/majorProject/api/v1.0/fetchTweets/<companyName>
Arguments:
companyName - fetching the tweets using company name

EXAMPLE: http://127.0.0.1:5000/majorProject/api/v1.0/fetchTweets/DaburIndia
Above URL fetches tweets for DaburIndia
"""
@app.route('/majorProject/api/v1.0/fetchTweets/<companyName>', methods=['GET'])
def fetching_tweets(companyName):
	tweets=[]
	t=[]
	file = open(companyName+'_tweets.csv',"w")
	csvWriter = csv.writer(file,delimiter=',')
	csvWriter.writerow(['Date','Tweet'])
	for status in tweepy.Cursor(api.user_timeline,screen_name=companyName).items(): #Fetching the tweets using screen_name(i.e Company Name)
		tweet = status.text  														#Extracting text from a tweet
		#tweet = preprocessing(tweet) #Preprocessing the tweets
		
		###Creating a tweets payload with date as the key and tweets posted on that date as value###
		if len(tweet) != 0:
			tweets.append([str(status.created_at).split()[0],tweet])

	###Storing the data in csv format###
	tweets=tweets[::-1]
	csvWriter.writerows(tweets)
	file.close()
	
	return jsonify({"Tweets Collection":"Success"})



"""
GET Reuqest URL: http://127.0.0.1:5000/majorProject/api/v1.0/getPolarity/<companyName>
Arguments:
companyName - access the CSV file of a company with past tweets

EXAMPLE: http://127.0.0.1:5000/majorProject/api/v1.0/getPolarity/DaburIndia
Above URL gets the polarity of tweets for DaburIndia
"""
@app.route('/majorProject/api/v1.0/getPolarity/<companyName>', methods=['GET'])
def get_polarity(companyName):
	model=Word2Vec.load("Sentiment Analysis Model") #Using the Word2Vec model
	with open('finalized_model.sav', 'rb') as f:
		forest = pickle.load(f)
	#forest = train_model(model,300)
	testDataVecs = calculate_average_feature_vector_test_data(model,300,companyName)
	test_model(forest, testDataVecs,companyName)

	return jsonify({"Polarity Collection":"Success"})




"""
GET Reuqest URL: http://127.0.0.1:5000/majorProject/api/v1.0/calculatePolarity/<companyName>
Arguments:
companyName - access the CSV file of a company with past tweets

EXAMPLE: http://127.0.0.1:5000/majorProject/api/v1.0/calculatePolarity/DaburIndia
Above URL calculates the polarity of tweets for DaburIndia
"""
@app.route('/majorProject/api/v1.0/calculatePolarity/<companyName>', methods=['GET'])
def calculate_polarity(companyName):
	desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 
	dataset=list(csv.reader(open(desktop+"/Git/Major-Project/Backend/"+companyName+"_output.csv")))
	dataset=dataset[1:]
	payload={}
	for i in range(len(dataset)):
		if dataset[i][1] not in payload:
			payload[dataset[i][1]] = {'pos':0,'neg':0}
		if int(dataset[i][2]) == 1:
			payload[dataset[i][1]]['pos']+=1
		elif int(dataset[i][2]) == 0:
			payload[dataset[i][1]]['neg']+=1


	file = open(companyName+'_sentiment.csv',"w")
	csvWriter = csv.writer(file,delimiter=',')
	csvWriter.writerow(['Date','Pos','Neg'])
	dates = list(payload.keys())
	dates.sort()
	prev=dates[0]
	dates=dates[1:]
	for date in dates:
		payload[date]['pos']+=payload[prev]['pos']
		payload[date]['neg']+=payload[prev]['neg']
		prev=date

	dates = list(payload.keys())
	dates.sort()
	for date in dates:
		csvWriter.writerow([date,payload[date]['pos'],payload[date]['neg']])
	file.close()


	return jsonify(payload)


if __name__ == "__main__":
	"""
	building_model()
	model=Word2Vec.load("Sentiment Analysis Model")
	forest = train_model(model,300)
	filename = 'finalized_model.sav'
	pickle.dump(forest, open(filename, 'wb'))
	"""
	app.run(debug = True) 	#Runs the Flask App, debug=True -> prints out prints out Python Errors(if any).



