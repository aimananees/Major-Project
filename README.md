# Major-Project


Setting Up Libraries
--------------------

(1) Make sure Python3 is installed.
	- To verify if Python3 is installed, run the following command:
		python3
	- To install Python3, do the following:
		sudo apt-get install software-properties-common
		sudo add-apt-repository ppa:deadsnakes/ppa
		sudo apt-get update
		sudo apt-get install python3.6
	- Verify if Python3 is installed properly by running the following command:
		python3 --version
	- Launch Python3 interpreter:
		python3
(2) Install pip and Flask
	- First install pip by running the following command:
		sudo apt-get update
		sudo apt-get install python3-pip
	- Install Flask by running the following command:
		sudo pip3 install flask

(3) Libraries required for running the Backend(Installation guide)
	- requests library
		sudo pip3 install requests
	- Tweepy library
		sudo pip3 install tweepy

(4)	Libraries required for Word2Vec
	- Gensim library
		sudo pip3 install --upgrade gensim

(5) Other data processing libraries such as
	- Pandas
		sudo pip3 install pandas
	- Numpy 
		sudo pip3 install numpy
	- Regex(re)
		sudo pip3 install regex
	- preprocessor
		sudo pip3 install tweet-preprocessor

(6) Library for Classification
	- Sklearn Library
		pip install -U scikit-learn

(7) Library for initialzing the Stacked Ensemble Regressor
    - mlxtend
        sudo pip3 install mlxtend


Running the Backend
-------------------

(1) To start the Flask App, run the following command:
		python3 Backend.py
(2) Open the browser and type the following to get information about a specific stock:
		http://127.0.0.1:5000/majorProject/api/v1.0/fetchStockData/<quandlCode>/<companyCode>/<startDate>/<endDate>
		Here:
			quandlCode - a specific code for the stock
			companyCode - NSE code for the stock
			startDate, endDate - fetching stock data between a start date and end date

		EXAMPLE: http://127.0.0.1:5000/majorProject/api/v1.0/fetchStockData/NSE/DABUR/01-01-2018/02-12-2018
(3) In order to fetch the polarity of tweets for a specific company type the following:
		http://127.0.0.1:5000/majorProject/api/v1.0/getPolarity/<companyName>
		Here:
			companyName/twitter-sceeen-name - fetching the tweets using company name
		
		EXAMPLE: http://127.0.0.1:5000/majorProject/api/v1.0/getPolarity/DaburIndia

(4) In order to calculate polarity of tweets for a specific company type the following:
		http://127.0.0.1:5000/majorProject/api/v1.0/calculatePolarity/<companyName>
		Here:
			companyName - access the CSV file of a company with past tweets

		EXAMPLE: http://127.0.0.1:5000/majorProject/api/v1.0/calculatePolarity/DaburIndia

(5) Run the machine learning module
        After creation of the price information (PI) and sentiment score (SS) csv files
        open the modularize.py file and edit lines 365 and 369 with the appropriate path to the PI.csv and SS.csv respectively    
        Use the given naming format to store results for easier analysis in the future in lines 372 and 373
            R^2 for Technical indicator only (x months) using SVM meta regressor: results/<Company Name>_<x>m_R2_technical_SVM_meta.csv
            R^2 for Technical indicator only (x months) using ExtRa meta regressor: results/<Company Name>_<x>m_R2_technical_Ext_meta.csv
            RMSE for Technical indicator only (x months) using SVM meta regressor: results/<Company Name>_<x>m_RMSE_technical_SVM_meta.csv
            RMSE for Technical indicator only (x months) using ExtRa meta regressor: results/<Company Name>_<x>m_RMSE_technical_Ext_meta.csv
            R^2 for Technical + sentiment data (x months) using SVM meta regressor: results/<Company Name>_<x>m_R2_sent_tech_SVM_meta.csv
            R^2 for Technical + sentiment data (x months) using ExtRa meta regressor: results/<Company Name>_<x>m_R2_sent_tech_Ext_meta.csv
            RMSE for Technical + sentiment data (x months) using SVM meta regressor: results/<Company Name>_<x>m_RMSE_sent_tech_SVM_meta.csv
            RMSE for Technical+ sentiment data (x months) using ExtRa meta regressor: results/<Company Name>_<x>m_RMSE_sent_tech_Ext_meta.csv
        
        Finally, Run the code using        
            python3 modularize.py

(6) Analyzing results
        Within the results folder, analyze.py can be edited on lines 3 and 4 to specify which files are to be compared.
        It finds the percentage difference between two quantities and averages the difference across all the rows. Prints the average.
        Run using
            python3 analyze.py
