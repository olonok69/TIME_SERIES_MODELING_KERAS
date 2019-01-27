import os
import sys
import numpy as np
import datetime
import itertools
sys.path.append('C:\\Program Files\\Anaconda3\\Lib\\site-packages')
sys.path.append('C:\\Program Files\\Continuum\\Anaconda3\\Lib\\site-packages')
#C:\Program Files\Anaconda3\Lib\site-packages\sklearn\neighbors
import matplotlib.pyplot as plt
import pandas as pd
import library as mio
from sklearn.metrics import classification_report



def write_Excel(__df_r1, filename, title):
	print ("Printing Report..."+ str(title))
	total_file = os.path.dirname(os.path.realpath(__file__)) + "\\raw_data\\" + filename
	writer = pd.ExcelWriter(total_file)

	__df_r1.to_excel(writer,title)
	writer.save()

def correlations(df):

	df_correlation=df.corr() # create correlation matrix

	#write_Excel(df_correlation, "Correlation Matrix.xlsx"," Correlation Matrix" )
	return

#NASDAQ Composite (^IXIC Yahoo Finance)
#Dow Jones Industrial Average (^DJI Quandl)
#Frankfurt DAX (^GDAXI Yahoo Finance)
#London FTSE-100 (^FTSE Yahoo Finance)
#Paris CAC 40 (^FCHI Yahoo Finance)
#Tokyo Nikkei-225 (^N225 Yahoo Finance)
#Hong Kong Hang Seng (^HSI Yahoo Finance)
#Australia ASX-200 (^AXJO Yahoo Finance)
	


def main():

	indices_list_Complete = ["^GSPC","SPY","^IXIC", "^DJI", "^GDAXI", "^FTSE","^FCHI", "^N225","^HSI", "^AXJO","ORB", "EUR","AUD","GBP","JPY", "SILVER", "GOLD", "PLAT","WT1010"] # reduced list only the most correlated
	indices_list_reduced = ["^GSPC","^IXIC", "^DJI", "^GDAXI", "^FTSE","^N225","^HSI", "^AXJO", "EUR","JPY"] # Indexes correlated >.5
	indices_list_ultra = ["^GSPC","^IXIC", "^DJI", "^GDAXI", "^FTSE","^N225"] # Indexes correlated >.7
	indices_day_after=["^GSPC", "^N225","^HSI", "^AXJO","ORB", "AUD","JPY", "GOLD", "PLAT"] # this index have closing price before NY stock exchange opens
	#index_list=["^GSPC"]
	algorithm=["KNN","RFC","SVM","ADA Bost","GTB","LDA", "SGD","LRC", "VOT", "DTC"]
	#algorithm=["SVM","ADA Bost"]
	#algorithm=["LRC"]
	optimiza=0 # Control to optize Algorithms or to Produce outcomes
	TEST=6# Tipo Ge feautures
	numDaysArray = [1] # day, week, month, quarter, year
	numDays = [5,21,63]
	param='Default'
	version='Completed'

	start_date = "2003-01-01"
	end_date = "2017-01-01"
	dates = pd.date_range(start_date, end_date)  # date range as index
	df_accu = mio.Load_DataFrames()

	indices_list = indices_list_Complete
 
	if indices_list == indices_list_ultra:
         version='Ultra'
	elif indices_list == indices_list_reduced:
         version='reduced' 
	elif indices_list == indices_list_Complete:
         version='Completed' 
	elif indices_list == indices_day_after:
         version='Day After' 

	kFOLDS=0  # use K-FOLDS  yes or no 1=YES, 0 = NO
 
	if kFOLDS==1:
         param='K-folds'
	else:
         param='Default'     

	df_index = mio.get_data(indices_list, dates) # get data from index and return a dataframe with all prices , but adjusted to the days we have prices of SP500
	#df_index=mio.shit_day(df_index,indices_list,indices_day_after)	# Move the closing price 1 day before if needed
	df_index.fillna(method='ffill', inplace=True)# fill Nan with previos value as order is ascending date 
	df_index.fillna(method='bfill', inplace=True)# fill NaN first day
	#write_Excel(df_index, "prices_shift.xlsx", "prices_shift") 	
	#Normalize data as have diferent dimension
	df_index_normalized=mio.normalize(df_index,indices_list)
	correlations(df_index_normalized) ##CORRELATIONS#
	#write_Excel(df_index_normalized, "prices_merged_normalized.xlsx", "Correlation") 
		
	df_adjusted=df_index_normalized.copy()
	
      # SP Adjust Price 1 + Rest Adjust Rolling Average 5,21,63
      # SP 500 adjusted return 1 + rest feuturest adjust return rolling 5,21,63
	
	if TEST==1 or TEST==5:
      # SP Adjust Price 1 + Rest Adjust Rolling Average 5,21,63
      # SP 500 adjusted return 1 + rest feuturest adjust return rolling 5,21,63
         features_taken='SP Adjust Price 1 + Rest Adjust Rolling Average 5' #TEST 1
         for symbol in indices_list:
             
             df_adjusted=mio.adjusted_return(df_adjusted,symbol, numDaysArray) #calculate day returns in day t -n days from Array numDaysArray
             df_adjusted=mio.remove_col(df_adjusted,symbol,'^GSPC_Adj_1') # delete not necessary columns
             
         df_adjusted_2=mio.Cleaning(df_adjusted) 
         df_adjusted_rolling=df_adjusted_2.copy()
         mylist=df_adjusted_rolling.columns.values.tolist()
             
	elif TEST==2:
      # SP Adjust Price 1 + Rest Adjust Rolling Average 5,21,63
      # SP 500 adjusted return 1 + rest feuturest adjust return rolling 5,21,63
         features_taken='SP Adjust Price  + Momentum 5' #TEST 1
         
         df_adjusted_rolling=df_adjusted.copy()
         mylist=df_adjusted_rolling.columns.values.tolist()

	elif TEST==3:
         features_taken='SP Adjust Price 1 + Rest Volatility Rolling Average 5' #TEST 3

         df_adjusted_rolling=df_adjusted.copy()
         mylist=df_adjusted_rolling.columns.values.tolist()

	elif TEST==4:
         features_taken='SP Adjust Price 1 + exponential moving weighted average with a span of 5' #TEST 4

         df_adjusted_rolling=df_adjusted.copy()
         mylist=df_adjusted_rolling.columns.values.tolist()

	elif TEST==6:
         features_taken='SP Adjust Price 1 + Combination of Series' #TEST 4

         df_adjusted_rolling=df_adjusted.copy()
         mylist=df_adjusted_rolling.columns.values.tolist()

	
	for symbol in mylist:	

         if TEST==1 or TEST==5: # Adjusted prices + adjusted prices rolling N days
             numDays = [5]          
             df_adjusted_rolling=mio.rolling_average(df_adjusted_rolling,symbol, numDays) #SP Adjust Price 1 + Rest Adjust Rolling Average 5,21,63
             df_adjusted_rolling=mio.remove_col(df_adjusted_rolling,symbol,'^GSPC_Adj_1') # delete not necessary columns
             df_adjusted_rolling=mio.Cleaning(df_adjusted_rolling) #(Remove, inf with NaN and replace Bfill) 

         if TEST==2: # Adjusted prices + Momentum rolling N days
             numDays = [5]         
             df_adjusted_rolling=mio.MOM(df_adjusted_rolling,symbol, numDays) #calculate momentum in n days from Array numDaysArray
             df_adjusted_rolling=mio.remove_col(df_adjusted_rolling,symbol,'^GSPC') # delete not necessary columns
             df_adjusted_rolling=mio.Cleaning(df_adjusted_rolling) #(Remove, inf with NaN and replace Bfill) 

         if TEST==3: # Adjusted prices + Volatility rolling N days
             numDaysArray = [5]
             df_adjusted_rolling=mio.Volatity_n(df_adjusted_rolling,symbol, numDaysArray) #calculate volatility in n days from numDaysArray 2, 21, 63
             df_adjusted_rolling=mio.remove_col(df_adjusted_rolling,symbol,'^GSPC') # delete not necessary columns
             df_adjusted_rolling=mio.Cleaning(df_adjusted_rolling) #(Remove, inf with NaN and replace Bfill) 

         if TEST==4: # Adjusted prices + EWM rolling N days
             numDays = [5]         
             df_adjusted_rolling=mio.ExpMovingAverage(df_adjusted_rolling,symbol, numDays) #exponential moving weighted average with a span of 5,21,63
             df_adjusted_rolling=mio.remove_col(df_adjusted_rolling,symbol,'^GSPC') # delete not necessary columns
             df_adjusted_rolling=mio.Cleaning(df_adjusted_rolling) #(Remove, inf with NaN and replace Bfill) 

         if TEST==6: # Adjusted prices + EWM rolling N days
             if symbol != '^GSPC':
                 df_adjusted_rolling=mio.adjusted_return(df_adjusted_rolling,symbol, numDaysArray)
                 numDays = [5]
                 name_symbol=symbol+"_Adj_1"
                 df_adjusted_rolling=mio.rolling_average(df_adjusted_rolling,name_symbol, numDays)             
             numDays = [21] 
             df_adjusted_rolling=mio.MOM(df_adjusted_rolling,symbol, numDays)
             numDays = [2] 
             df_adjusted_rolling=mio.Volatity_n(df_adjusted_rolling,symbol, numDays) 
             numDays = [21]              
             df_adjusted_rolling=mio.ExpMovingAverage(df_adjusted_rolling,symbol, numDays) #exponential moving weighted average with a span of 5,21,63
             df_adjusted_rolling=mio.remove_col(df_adjusted_rolling,symbol,'^GSPC') # delete not necessary columns
             df_adjusted_rolling=mio.Cleaning(df_adjusted_rolling) #(Remove, inf with NaN and replace Bfill) 


	if TEST!=1 and TEST!=5 :

         M = pd.Series(df_adjusted_rolling["^GSPC"].pct_change(1).shift(-1), name = "^GSPC_Adj_" + str(1))  
         
         df_adjusted_rolling = df_adjusted_rolling.join(M)
         del df_adjusted_rolling['^GSPC']

	#write_Excel(df_adjusted_rolling, "prices_adjusted_rolling.xlsx", "rolling")
	df_adjusted_final=mio.Cleaning(df_adjusted_rolling) #(Remove, inf with NaN and replace Bfill) 
	#write_Excel(df_adjusted_final, "prices_adjusted_rolling_test6.xlsx", "Test6")
	#pie =df_adjusted_final.plot(figsize=(15,20), subplots=True, grid=True, layout=(20,4))
	#[ax.legend(loc=5) for ax in plt.gcf().axes]
	#plt.tight_layout()
	#fig = pie[0].get_figure()
	#fig.savefig("TEST6.jpg") 
	df_final, X_train, y_train, X_test, y_test=mio.prepareDataForClassification(df_adjusted_final,'01/01/2013  00:00:00', TEST)
	#write_Excel(df_final, "prices_final.xlsx", "Data Final")
 
 
        # test algorithm. GO OVER ALL ALGORITHMS with other without Kfols
	if optimiza ==0:
		for alg in algorithm:
		 
			 if kFOLDS==0:
				 score, f1score, y_test, predictions, training, predecir,roc,mat=mio.call_alg(X_train, y_train, X_test, y_test, alg)
				 report = classification_report(y_test, predictions)
				 mio.classifaction_report_csv(report,alg,version,features_taken,kFOLDS,score, training, predecir,roc,mat)


			 elif kFOLDS==2:# WALF FORWARD VALIDATION
				 
				 score, f1score, y_test, predictions, training, predecir,roc,mat=mio.Walk_Forward_Validation_CV(df_adjusted_final,'01/01/2013  00:00:00', alg)
				 report = classification_report(y_test, predictions)
				 mio.classifaction_report_csv(report,alg,version,features_taken,kFOLDS,score, training, predecir,roc,mat)

			 else:
				 nfolds=2
				 score, f1score, y_test, predictions, training, predecir,roc,mat=mio.TimeSeriesCrossValidation(X_train, y_train, nfolds, alg)
				 report = classification_report(y_test, predictions)
				 if kFOLDS==1:
				          param1=param+ " " + str(nfolds)
                            
				 mio.classifaction_report_csv(report,alg,version,features_taken,param1,score, training, predecir,roc,mat)
     
			 if kFOLDS==1 and alg=="KNN" :        
				 param=param+ " " + str(nfolds)
			
			# Keep track of the TEST
			 df_accu = df_accu.append({'Algorithm': alg, 'Accuracy': score, 'Parameters': param , 'version':version, 'F1Score': f1score,'Feautures': features_taken}, ignore_index=True)
			 print(classification_report(y_test, predictions))
			 #print(y_test)
			 #print(predictions)    

	if optimiza ==1:
		#parameters=mio.GSSVC(X_train, y_train, X_test, y_test, TEST)
		#print(parameters)
		#parameters=mio.GSKNN(X_train, y_train, X_test, y_test, TEST)
		#print(parameters)
		#parameters=mio.GSRFC(X_train, y_train, X_test, y_test, TEST)
		#print(parameters)
		#parameters = mio.GSGTB(X_train, y_train, X_test, y_test, TEST)
		#print(parameters)
		#parameters = mio.GSLRC(X_train, y_train, X_test, y_test, TEST)
		#print(parameters)
		#parameters = mio.GSADA(X_train, y_train, X_test, y_test, TEST)
		#print(parameters)
		#parameters = mio.GSLDA(X_train, y_train, X_test, y_test, TEST)
		#print(parameters)
		#parameters = mio.GSSGD(X_train, y_train, X_test, y_test, TEST)
		#print(parameters)
		#parameters = mio.GSDTC(X_train, y_train, X_test, y_test, TEST)
		#print(parameters)
		parameters, parameters1 =mio.CVperformEnsemberBlendingClass(X_train, y_train, X_test, y_test)
		print(parameters)
		print(parameters1)
	if optimiza ==2:
		df_out, accuracy, Sscore, accuracy2, Sscore2=mio.performEnsemberBlendingClass(X_train, y_train, X_test, y_test)  
		path = os.path.dirname(os.path.realpath(__file__))
		print ("Loading csv...")
		file_path = path +  "\\raw_data\\binary.csv"
		df_out.to_csv(file_path, sep=',')
		df_accu = df_accu.append({'Algorithm': "BLE1", 'Accuracy': accuracy, 'Parameters': param , 'version':version, 'F1Score': Sscore,'Feautures': features_taken}, ignore_index=True)
		df_accu = df_accu.append({'Algorithm': "BLE2", 'Accuracy': accuracy2, 'Parameters': param , 'version':version, 'F1Score': Sscore2,'Feautures': features_taken}, ignore_index=True)


	if optimiza ==0 or optimiza ==2:
		write_Excel(df_accu,"Accuracy.xlsx","Accuraccy")



	
	#write_Excel(df_adjusted_rolling, "prices_adjusted_rolling.xlsx", "Momentum")
	#write_Excel(df_adjusted_rolling, "prices_rollings.xlsx")
	#write_Excel(df_index, "prices_merged.xlsx")
	#write_Excel(df_index_normalized, "prices_merged_normalized.xlsx")
	
                

main()
