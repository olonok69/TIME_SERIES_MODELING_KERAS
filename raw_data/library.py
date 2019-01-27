import pandas
import numpy as np
#from sklearn import preprocessing
#from sklearn import svm
#from sklearn import cross_validation
#import pandas_datareader.data as web
#import quandl

daysAhead = 270

def get_web_data(symbols, dates):
    #"""Read stock data (adjusted close) for given symbols from CSV files."""
	df_final = pd.DataFrame(index=dates)
	i=0
	fechastart='2007-01-01'
	fechaend='2017-01-01'
                
	for symbol in symbols:
                               
		df_temp = web.DataReader(symbol, start=fechastart, end=fechaend,data_source='yahoo')["Date", "Adj Close"]
		df_temp = df_temp.rename(columns={"Adj Close": symbol})
		df_final = df_final.join(df_temp)
		i+=1
		if i == 1:  # drop dates SPY did not trade
			df_final = df_final.dropna(subset=[symbol])
	return df_final

	
def get_web_quandl(symbol, simbolo):

	df =  quandl.get(symbol, trim_start = "01/01/2007", trim_end ="01/01/2017", authtoken="_N85bWLCNCWz14smKHSi")
	name=simbolo
	df.columns.values[-1] = 'AdjClose'
	df.columns = df.columns + '_' + name
	df['Return_%s' %name] = df['AdjClose_%s' %name].pct_change()
	return df

def calcPriceVolatility(numDays, priceArray):

	global daysAhead

	# make price volatility array

	volatilityArray = []

	movingVolatilityArray = []

	for i in range(1, numDays+1):

		percentChange = 100 * (priceArray[i] - priceArray[i-1]) / priceArray[i-1]

		movingVolatilityArray.append(percentChange)

	volatilityArray.append(np.mean(movingVolatilityArray))

	for i in range(numDays + 1, len(priceArray) - daysAhead):

		del movingVolatilityArray[0]

		percentChange = 100 * (priceArray[i] - priceArray[i-1]) / priceArray[i-1]

		movingVolatilityArray.append(percentChange)

		volatilityArray.append(np.mean(movingVolatilityArray))



	return volatilityArray

# calculate momentum array

def calcMomentum(numDays, priceArray):

	#global daysAhead

	# now calculate momentum

	momentumArray = []

	movingMomentumArray = []

	for i in range(1, numDays + 1):

		movingMomentumArray.append(1 if priceArray[i] > priceArray[i-1] else -1)

	momentumArray.append(np.mean(movingMomentumArray))

	for i in range(numDays+1, len(priceArray) - daysAhead):

		del movingMomentumArray[0]

		movingMomentumArray.append(1 if priceArray[i] > priceArray[i-1] else -1)

		momentumArray.append(np.mean(movingMomentumArray))



	return momentumArray