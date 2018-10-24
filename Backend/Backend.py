from flask import Flask,jsonify
import requests
import csv
app = Flask(__name__)



@app.route('/majorProject/api/v1.0/fetchStockData/<quandlCode>/<companyCode>/<startDate>/<endDate>', methods=['GET'])
def fetchStockData(quandlCode, companyCode, startDate, endDate):

	URL="https://www.quandl.com/api/v3/datasets/"+quandlCode+"/"+companyCode+"?start_date="+startDate+"&end_date="+endDate+"&api_key=5kxvD2BwBUYj1aTcMnj5"
	fetchStockData = requests.get(url=URL)
	fetchStockData=fetchStockData.json()

	with open(fetchStockData['dataset']['name']+'.csv',"w") as file:
		csvWriter = csv.writer(file,delimiter=',')
		csvWriter.writerow(['Date','Open','High','Low','Last','Close','Total Trade Quantity','Turnover (Lacs)'])
		csvWriter.writerows(fetchStockData['dataset']['data'])


	return jsonify({'name':fetchStockData['dataset']['name']},{'column_names':fetchStockData['dataset']['column_names']},{'data':fetchStockData['dataset']['data']})



if __name__ == "__main__":
	app.run(debug = True)