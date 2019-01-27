import os
import sys
import pandas as pd
import numpy as np
import time
import datetime
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.pylab as py
#from sklearn import svm
#from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

#import pandas_datareader.data as web
#import quandl
global kFOLDS
daysAhead = 270
kFOLDS=0

def get_data(symbols, dates):
    #"""Read stock data (adjusted close) for given symbols from CSV files."""
	df_final = pd.DataFrame(index=dates)
	i=0
	for symbol in symbols:
		path = os.path.dirname(os.path.realpath(__file__))
		
		file_path = path +  "\\raw_data\\" + symbol + ".csv"
		
		#print ("Loading csv..." + str(file_path))
		if symbol=="ORB"or symbol=="WT1010":#OIL
			df_temp = pd.read_csv(file_path, parse_dates=True, index_col="Date",usecols=["Date", "Value"], na_values=["nan"])
			df_temp = df_temp.rename(columns={"Value": symbol})
		elif symbol=="EUR" or symbol=="GBP" or symbol=="AUD" or symbol=="JPY":
			df_temp = pd.read_csv(file_path, parse_dates=True, index_col="DATE",usecols=["DATE", "RATE"], na_values=["nan"])
			df_temp = df_temp.rename(columns={"RATE": symbol})
		elif symbol=="PLAT":
			df_temp = pd.read_csv(file_path, parse_dates=True, index_col="Date",usecols=["Date", "London 08:00"], na_values=["nan"])
			df_temp = df_temp.rename(columns={"London 08:00": symbol})
		elif symbol=="GOLD":
			df_temp = pd.read_csv(file_path, parse_dates=True, index_col="Date",usecols=["Date", "USD (AM)"], na_values=["nan"])
			df_temp = df_temp.rename(columns={"USD (AM)": symbol})
		elif symbol=="SILVER":
			df_temp = pd.read_csv(file_path, parse_dates=True, index_col="Date",usecols=["Date", "USD"], na_values=["nan"])
			df_temp = df_temp.rename(columns={"USD": symbol})
		else:
			df_temp = pd.read_csv(file_path, parse_dates=True, index_col="Date",usecols=["Date", "Adj Close"], na_values=["nan"])
			df_temp = df_temp.rename(columns={"Adj Close": symbol})
		
		#df_temp = df_temp.rename(columns={"Adj Close": symbol})
		df_final = df_final.join(df_temp)
		i+=1
		if i == 1:  # drop dates SPY did not trade
			df_final = df_final.dropna(subset=[symbol])

	return df_final

def write_Ex(__df_r1, filename, title):
	print ("Printing Report..."+ str(title))
	total_file = os.path.dirname(os.path.realpath(__file__)) + "\\raw_data\\" + filename
	writer = pd.ExcelWriter(total_file)

	__df_r1.to_excel(writer,title)
	writer.save()


def plot_confusion_matrix(cm,alg, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    title=title+" " +alg
    plt.title(title)
    plt.colorbar()

    names=["Up","Down"]
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names)
    plt.yticks(tick_marks, names)
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_cm(cm, alg):
    np.set_printoptions(precision=2)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 4))
    plot_confusion_matrix(cm_normalized, alg)
    file = os.path.dirname(os.path.realpath(__file__)) + "\\out\\" +"CM_"+alg+".png"

    plt.savefig(file)
    
def shift_day(df_index,indices_list,indices_day_after):
    for indice in indices_list:
        if indice not in str(indices_day_after):
            df_index[indice] = df_index[indice].shift(1)

    return df_index
    
def Load_DataFrames():

	# Open Excel files to load them in a dataframe
	path = os.path.dirname(os.path.realpath(__file__))
	file_Accuracy = path + "\\raw_data\\Accuracy.xlsx"

		
	# Load files in a dataframe for manipulation
	accu = pd.ExcelFile(file_Accuracy)

	# Select first sheet in the file 
	_df_accu = accu.parse(accu.sheet_names[0])

	return _df_accu

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

def adjusted_return(df, symbol,numDaysArray):
	#df_final = pd.DataFrame(index=dates)
	for n in numDaysArray:
     #["^GSPC","SPY","^IXIC", "^DJI", "^GDAXI", "^FTSE","^FCHI",  "EUR","GBP", "SILVER", "WT1010"] 
         if symbol =="^GSPC"or symbol =="^IXIC"or symbol =="^DJI"or symbol == "^GDAXI"or symbol =="^FTSE"or symbol == "^FCHI"or symbol =="EUR"or symbol == "GBP"or symbol == "SILVER" or symbol == "WT1010":
             M = pd.Series(df[symbol].pct_change(n).shift(-1), name = str(symbol)+ '_Adj_' + str(n)) 
             #M = pd.Series(df[symbol].pct_change(n), name = str(symbol)+ '_Adj_' + str(n))              
         else: 
             M = pd.Series(df[symbol].pct_change(n), name = str(symbol)+ '_Adj_' + str(n)) 
         df = df.join(M)
	return df

def rolling_average(df, symbol,numDaysArray): 
	for n in numDaysArray:
		if symbol=="^GSPC_Adj_1":
			M=pd.Series(df[str(symbol)].rolling(window = n, center = False).mean().shift(1), name = str(symbol)+ '_Roll_Avg_' + str(n))
		else:
			M=pd.Series(df[str(symbol)].rolling(window = n, center = False).mean(), name = str(symbol)+ '_Roll_Avg_' + str(n))
		df = df.join(M)
	return df
 

def remove_col(df, symbol, symbol2):
	if symbol != symbol2:
		del df[symbol]
	return df

def normalize(df, symbols):
	result = df.copy()
	for symbol in df.columns:
		max_value = df[symbol].max()
		min_value = df[symbol].min()
		
		result[symbol] = (df[symbol] - min_value) / (max_value - min_value)
	return result


	
def mergeDataframes(out, dataset1,dataset2, symbol1,symbol2):
	out= pd.merge(dataset1, dataset2, on='Date', how='left')
	return out

def Volatity_n(df, symbol,numDaysArray):
	#numDaysArray = [2, 21, 63]
	#numDaysArray = [2]
	for n in numDaysArray:
	#M = pd.Series(df[symbol].pct_change(n).std(), name = 'Volatility_' + str(n)) 
             #if symbol != "^GSPC":
             df["Log_Ret"] = np.log(df[symbol] / df[symbol].shift(1))
             df[str(symbol)+ "_Vol_"+ str(n)] = df["Log_Ret"].rolling(window=n, center=False).std() * np.sqrt(252) # if we multiply sqrt(252) is anualized volatility
             del df["Log_Ret"]
	return df 

def MOM(df, symbol,numDaysArray): 
	for n in numDaysArray:
         #if symbol != "^GSPC":
         M = pd.Series(df[symbol].diff(n), name = str(symbol)+ '_MoM_' + str(n))  
         df = df.join(M)
	return df  
	
	
def ExpMovingAverage(df, symbol,numDaysArray):
	#ts_log = np.log(df[str(symbol)])
	for n in numDaysArray:
         #if symbol !="^GSPC":
         M = pd.Series(df[symbol].ewm(span=n).mean(), name = str(symbol)+ '_EWA_' + str(n))
         df = df.join(M)
         #df[str(symbol)+ "_EMA_"+ str(n)] = pd.ewma(df[symbol], span=n, freq="D")
		
         #Series.ewm(min_periods=0,adjust=True,ignore_na=False,span=1,freq=D).mean()	
	return df

def Label_Change (row):
	if row['^GSPC_Adj_1'] > 0 :
		return 'Up'

	return 'Down'

def Label_Change2 (row):
	if row['Real'] > 0 :
		return 'Up'

	return 'Down'
	
def Cleaning(df): #(Remove, inf with NaN and replace Bfill) 
	df.replace([np.inf, -np.inf], np.nan)
	df[df==np.inf] = np.nan
	df[df==-np.inf] = np.nan
	df.fillna(method='bfill', inplace=True)
	df.fillna(method='ffill', inplace=True)
	return df
 
def plotvalidation_curve(y_test,predictions):
	fig, ax = plt.subplots()
	ax.scatter(y_test, predictions)
	ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	plt.show()
	return
    
def prepareDataForClassification(dataset, start_test, test):

	le = preprocessing.LabelEncoder()
    
	dataset['UpDown'] = dataset.apply (lambda row: Label_Change (row),axis=1)
	dataset1=dataset.truncate(before='2003-07-01') #delete all values up to the first rolling average 63 days
    #dataset.UpDown[dataset.UpDown >= 0] = 'Up'
    #dataset.UpDown[dataset.UpDown < 0] = 'Down'
	dataset1.UpDown = le.fit(dataset1.UpDown).transform(dataset1.UpDown)
	if test==2:    
         features = dataset1.columns[0:-1]
         X = dataset1[features]  
         del X["^GSPC_Adj_1"]  
	if test==1 or test==5:    
         features = dataset1.columns[0:-1]
         X = dataset1[features]  
         del X["^GSPC_Adj_1"]  
	if test==3:    
         features = dataset1.columns[0:-1]
         X = dataset1[features]  
         del X["^GSPC_Adj_1"]  
	if test==4:    
         features = dataset1.columns[0:-1]
         X = dataset1[features]  
         del X["^GSPC_Adj_1"] 
	if test==6:    
         features = dataset1.columns[0:-1]
         X = dataset1[features]  
         del X["^GSPC_Adj_1"] 
         
	y = dataset1.UpDown    
    
	X_train = X[X.index < start_test]
	y_train = y[y.index < start_test]              
    
	X_test = X[X.index >= start_test]    
	y_test = y[y.index >= start_test]

	#path = os.path.dirname(os.path.realpath(__file__))


	#file_path = path +  "\\raw_data\\" + "y_train.csv"

	#y_train.to_csv(file_path, index = False)
    
	#file_path = path +  "\\raw_data\\" + "y_testn.csv"   
	#y_test.to_csv(file_path, index = False)

     
	#write_Ex(X_train, "X_train.xlsx", "X_train")

	#write_Ex(X_test, "X_test.xlsx", "X_test") 

    
	return X, X_train, y_train, X_test, y_test   
	

def performEnsemberBlendingClass(X_train, y_train, X_test, y_test):#, parameters, fout, savemodel):
	print("Ensembler Blending Classification")
	start_date = "2013-01-01"
	end_date = "2017-01-01"
	dates = pd.date_range(start_date, end_date)	
	df_out = pd.DataFrame(index=dates)
	df_out["Real"] = y_test.copy()
	df_out=df_out.dropna() # so we can modify it

	le = preprocessing.LabelEncoder()
    
	df_out['UpDown'] = df_out.apply (lambda row: Label_Change2 (row),axis=1)
	df_out.UpDown = le.fit(df_out.UpDown).transform(df_out.UpDown) 
   
	for i in range(50):   
         clf1 = SVC(C=((i+1)*100)+3200, kernel= 'sigmoid', gamma=0.05) #'C': 3500, 'kernel': 'sigmoid', 'gamma': 0.05
         clf2 =  RandomForestClassifier(n_estimators=4000, criterion= 'gini') #'criterion': 'gini', 'n_estimators': 4000}
         clf3 = SGDClassifier(penalty='elasticnet', loss='perceptron', learning_rate= 'invscaling', eta0=0.1, alpha=0.001)
                             #{'penalty': 'elasticnet', 'eta0': 0.1, 'alpha': 0.001, 'loss': 'perceptron', 'learning_rate': 'invscaling'}
         clf4 = AdaBoostClassifier(n_estimators=200, algorithm='SAMME')
         clf5= GradientBoostingClassifier(min_samples_leaf=75, n_estimators=90+(i*10), max_features= 'auto',min_samples_split=300, learning_rate=0.1)
         clf6 = neighbors.KNeighborsClassifier(algorithm='ball_tree', n_neighbors=350, weights='distance', leaf_size= 30)  
         clf7 = LinearDiscriminantAnalysis(store_covariance=True, n_components= 1, solver= 'lsqr', shrinkage= 0.1)
         
         print("Fitting")
         print("SVC"+str(i))         
         clf1.fit(X_train, y_train)
         print("RFC"+str(i))
         clf2.fit(X_train, y_train)
         print("SGD"+str(i))
         clf3.fit(X_train, y_train)
         print("ADA"+str(i))
         clf4.fit(X_train, y_train)
         print("GTB"+str(i))
         clf5.fit(X_train, y_train)
         print("KNN"+str(i))
         clf6.fit(X_train, y_train)         
         print("LDA"+str(i))
         clf7.fit(X_train, y_train)  
         
         print("predicting")	

  
         df_out['SVM'+str(i)] = clf1.predict(X_test)
         df_out['RFC'+str(i)] = clf2.predict(X_test)
         df_out['SDG'+str(i)] = clf3.predict(X_test)
         df_out['ADA'+str(i)] = clf4.predict(X_test)
         df_out['GTB'+str(i)] = clf5.predict(X_test)
         df_out['KNN'+str(i)] = clf6.predict(X_test)
         df_out['LDA'+str(i)] = clf7.predict(X_test)

	features = df_out.columns[1:]
	X = df_out[features] 
	del X["UpDown"]  
	#print(X)
	y = df_out.UpDown

	start_test="2016-01-01"   
	#print(y)		    
	X_train = X[X.index < start_test]
	y_train = y[y.index < start_test]              
  	  
	X_test = X[X.index >= start_test]    
	y_test = y[y.index >= start_test]
    #{'kernel': 'rbf', 'C': 100.0, 'gamma': 0.1}
    #{'penalty': 'l2', 'C': 0.001, 'solver': 'newton-cg'}
    
	clfsb= LogisticRegression(penalty= 'l2', C= 0.001, solver= 'newton-cg')   
	clfsb.fit(X_train, y_train)
	clfsb2= SVC(C=100,kernel= 'rbf', gamma=0.1)   
	clfsb2.fit(X_train, y_train)
	#training=end-start
	accuracy = clfsb.score(X_test, y_test)
	accuracy2 = clfsb2.score(X_test, y_test)
	#start = datetime.datetime.now()
	predictions = clfsb.predict(X_test)
	predictions2 = clfsb2.predict(X_test)
	#end = datetime.datetime.now()
	Sscore = f1_score(y_test, predictions)
	Sscore2 = f1_score(y_test, predictions2)

	#predecir=end-start
	print("F1 Score Blender " +str(Sscore))
	print("Accuracy Blender " +str(accuracy))
	print("F1 Score Blender 2 " +str(Sscore2))
	print("Accuracy Blender 2 " +str(accuracy2))
	cm = confusion_matrix(y_test, predictions)
	plot_cm(cm,"Blender1 Classification")       
	cm1 = confusion_matrix(y_test, predictions2)
	plot_cm(cm1,"Blender2 Classification")   
	return df_out, accuracy, Sscore, accuracy2, Sscore2

def CVperformEnsemberBlendingClass(X_train, y_train, X_test, y_test):#, parameters, fout, savemodel):
	def f1_score_on_test(estimator, x, y):
		y_pred = estimator.predict(X_test)
		return f1_score(y_test.values, y_pred)
    
	print("Ensembler Blending Classification")
	start_date = "2013-01-01"
	end_date = "2017-01-01"
	dates = pd.date_range(start_date, end_date)	
	df_out = pd.DataFrame(index=dates)
	df_out["Real"] = y_test.copy()
	df_out=df_out.dropna() # so we can modify it

	le = preprocessing.LabelEncoder()
    
	df_out['UpDown'] = df_out.apply (lambda row: Label_Change2 (row),axis=1)
	df_out.UpDown = le.fit(df_out.UpDown).transform(df_out.UpDown) 
   
	for i in range(2):   
         clf1 = SVC(C=((i+1)*100)+3200, kernel= 'sigmoid', gamma=0.05) #'C': 3500, 'kernel': 'sigmoid', 'gamma': 0.05
         clf2 =  RandomForestClassifier(n_estimators=4000, criterion= 'gini') #'criterion': 'gini', 'n_estimators': 4000}
         clf3 = SGDClassifier(penalty='elasticnet', loss='perceptron', learning_rate= 'invscaling', eta0=0.1, alpha=0.001)
                             #{'penalty': 'elasticnet', 'eta0': 0.1, 'alpha': 0.001, 'loss': 'perceptron', 'learning_rate': 'invscaling'}
         clf4 = AdaBoostClassifier(n_estimators=200, algorithm='SAMME')
         clf5= GradientBoostingClassifier(min_samples_leaf=75, n_estimators=90+(i*10), max_features= 'auto',min_samples_split=300, learning_rate=0.1)
         clf6 = neighbors.KNeighborsClassifier(algorithm='ball_tree', n_neighbors=350, weights='distance', leaf_size= 30)  
         clf7 = LinearDiscriminantAnalysis(store_covariance=True, n_components= 1, solver= 'lsqr', shrinkage= 0.1)
         
         print("Fitting")
         print("SVC"+str(i))         
         clf1.fit(X_train, y_train)
         print("RFC"+str(i))
         clf2.fit(X_train, y_train)
         print("SGD"+str(i))
         clf3.fit(X_train, y_train)
         print("ADA"+str(i))
         clf4.fit(X_train, y_train)
         print("GTB"+str(i))
         clf5.fit(X_train, y_train)
         print("KNN"+str(i))
         clf6.fit(X_train, y_train)         
         print("LDA"+str(i))
         clf7.fit(X_train, y_train)  
         
         print("predicting")	

  
         df_out['SVM'+str(i)] = clf1.predict(X_test)
         df_out['RFC'+str(i)] = clf2.predict(X_test)
         df_out['SDG'+str(i)] = clf3.predict(X_test)
         df_out['ADA'+str(i)] = clf4.predict(X_test)
         df_out['GTB'+str(i)] = clf5.predict(X_test)
         df_out['KNN'+str(i)] = clf6.predict(X_test)
         df_out['LDA'+str(i)] = clf7.predict(X_test)

	features = df_out.columns[1:]
	X = df_out[features] 
	del X["UpDown"]  
	#print(X)
	y = df_out.UpDown

	start_test="2016-01-01"   
	#print(y)		    
	X_train = X[X.index < start_test]
	y_train = y[y.index < start_test]              
  	  
	X_test = X[X.index >= start_test]    
	y_test = y[y.index >= start_test]
	#clfsb= LogisticRegression(penalty= 'l1', C= 1, solver= 'liblinear')   
	#clfsb.fit(X_train, y_train)
	#clfsb2= SVC(C=0.01,kernel= 'rbf')   
	#clfsb2.fit(X_train, y_train)
	#training=end-start
	params_map = [{ 'C': [0.001,0.005, 0.01,0.02, 0.1,0.4, 0.5,0.6,0.7, 1.0, 2.0, 5.0, 10.0, 100.0],'kernel': ['rbf'],'gamma': ['auto',0.001,0.01, 0.05,0.1, 0.5, 1.0,4.0, 5.0, 6.0, 10.0,50.0,100.0,150.0]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000, 1500, 2000, 2500,3000, 3500,4000]}]#,
    #{'kernel': ['poly'], 'C': [1, 10, 100, 1000, 1500, 2000, 2500,3000, 3500,4000],'degree':[2,3,4,5,6,7,8,9,10,15], 'gamma': ['auto',0.001,0.01, 0.05,0.1, 0.5, 1.0, 5.0]},
    #{'kernel': ['sigmoid'], 'C': [1, 10, 100, 1000, 1500, 2000, 2500,3000, 3500,4000], 'gamma': ['auto',0.001,0.01, 0.05,0.1, 0.5, 1.0, 5.0]}]


	clf = GridSearchCV(SVC(), params_map, scoring=f1_score_on_test, verbose=100)


	test=5
	clf.fit(X_train, y_train)
    #joblib.dump(clf.best_estimator_, file_path, compress = 3)
	df=pd.DataFrame(clf.grid_scores_)
	total_file = os.path.dirname(os.path.realpath(__file__)) + "\\CV\\CVBlender_SVC_" + str(test)+ ".xlsx" 
	writer = pd.ExcelWriter(total_file)
	df.to_excel(writer,"SVC")
	writer.save() 			

	param_grid = [{'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty' :["l1"], 'solver' :["liblinear"] },
                  {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty' :["l2"], 'solver' :[ "newton-cg", "lbfgs", "sag"] }]

	clf1 = GridSearchCV(LogisticRegression(), param_grid, scoring=f1_score_on_test, verbose=100)
	clf1.fit(X_train, y_train)
	df=pd.DataFrame(clf1.grid_scores_)
	total_file = os.path.dirname(os.path.realpath(__file__)) + "\\CV\\CVBlenderLRC_" + str(test)+ ".xlsx"  
	writer = pd.ExcelWriter(total_file)
	df.to_excel(writer,"LRC")
	writer.save()	

	return clf.best_params_, clf1.best_params_




def performKNNClass(X_train, y_train, X_test, y_test):#, parameters, fout, savemodel):

	print("KNN binary Classification")
 #{{'leaf_size': 30, 'algorithm': 'ball_tree', 'n_neighbors': 350, 'weights': 'distance'}
	clf = neighbors.KNeighborsClassifier(algorithm='ball_tree', n_neighbors=300, weights='distance', leaf_size= 30)   	
	#clf = neighbors.KNeighborsClassifier()  
	start = datetime.datetime.now()

	clf.fit(X_train, y_train)
	end = datetime.datetime.now()
	training=end-start
	accuracy = clf.score(X_test, y_test)
	start = datetime.datetime.now()
	predictions = clf.predict(X_test)
	end = datetime.datetime.now()
	Sscore = f1_score(y_test, predictions)
	roc=mat=0
	if kFOLDS != 2:
         roc = roc_auc_score(y_test, predictions)
         mat = matthews_corrcoef(y_test, predictions) 
	predecir=end-start
	#cm = confusion_matrix(y_test, predictions)
	#plot_cm(cm,"KNN Binary Classification") 
	print("F1 Score KNN " +str(Sscore))
	return accuracy, Sscore, y_test,predictions  , training, predecir,roc,mat

def performRFClass(X_train, y_train, X_test, y_test):#, parameters, fout, savemodel):

	print("Random Forest Binary Classification")

	clf = RandomForestClassifier(n_estimators=500, criterion= 'gini') 
	#clf = RandomForestClassifier()# '{'n_estimators': 500, 'criterion': 'gini'}
	start = datetime.datetime.now()
	clf.fit(X_train, y_train)
	end = datetime.datetime.now()
	training=end-start
	accuracy = clf.score(X_test, y_test)
	start = datetime.datetime.now()
	predictions = clf.predict(X_test)
	end = datetime.datetime.now()
	Sscore = f1_score(y_test, predictions)
	roc=mat=0
	if kFOLDS != 2:
         roc = roc_auc_score(y_test, predictions)
         mat = matthews_corrcoef(y_test, predictions) 
	predecir=end-start
	#cm = confusion_matrix(y_test, predictions)
	#plot_cm(cm,"Random Forest Binary")  
	print("F1 Score RFC " +str(Sscore))

	return accuracy, Sscore, y_test,predictions , training, predecir,roc,mat

 
def performSGDClass(X_train, y_train, X_test, y_test):#, parameters, fout, savemodel):

	print("Stochastic Gradient Descent binary Classification")


	#{'learning_rate': 'invscaling', 'alpha': 0.01, 'loss': 'perceptron', 'penalty': 'elasticnet', 'eta0': 0.2}
	clf = SGDClassifier( alpha= 0.01,penalty='elasticnet', loss='perceptron', learning_rate= 'invscaling', eta0=0.2)
	#clf = SGDClassifier()
	start = datetime.datetime.now()
	clf.fit(X_train, y_train)
	end = datetime.datetime.now()
	training=end-start
	accuracy = clf.score(X_test, y_test)
	start = datetime.datetime.now()
	predictions = clf.predict(X_test)
	end = datetime.datetime.now()
	Sscore = f1_score(y_test, predictions)
	roc=mat=0
	if kFOLDS != 2:
         roc = roc_auc_score(y_test, predictions)
         mat = matthews_corrcoef(y_test, predictions) 
	predecir=end-start
	#cm = confusion_matrix(y_test, predictions)
	#plot_cm(cm,"Stochastic Gradient Descent") 
	print("F1 Score SGD " +str(Sscore))
	return accuracy, Sscore, y_test,predictions , training, predecir,roc,mat

def performSVMClass(X_train, y_train, X_test, y_test):#, parameters, fout, savemodel):
#"""
	print("SVM binary Classification")

	#print(n_components)
 	#c = parameters[0]
	#g =  parameters[1]
	#{{'C': 3500, 'kernel': 'sigmoid', 'gamma': 0.05}
      #{'kernel': 'rbf', 'C': 0.1, 'gamma': 100.0}

	clf = SVC(C=2000,kernel= 'linear')

	start = datetime.datetime.now()
	clf.fit(X_train, y_train)
	end = datetime.datetime.now()
	training=end-start
	accuracy = clf.score(X_test, y_test)
	start = datetime.datetime.now()
	#df_out['SVM'] = clf.predict(X_test) 
	predictions = clf.predict(X_test)
	end = datetime.datetime.now()
	Sscore = f1_score(y_test, predictions,average='binary')
	#cm = confusion_matrix(y_test, predictions)
	#plot_cm(cm,"Support Vector Machines")
	roc=mat=0
	if kFOLDS != 2:
         roc = roc_auc_score(y_test, predictions)
         mat = matthews_corrcoef(y_test, predictions) 
	predecir=end-start
	#write_Ex(df_out, "df_out.xlsx", "df_out") 
	#plotvalidation_curve(y_test,predictions)
	print("F1 Score SVM " +str(Sscore))
	return accuracy, Sscore, y_test,predictions, training, predecir,roc,mat
	
def performSVM_PCAClass(X_train, y_train, X_test, y_test):#, parameters, fout, savemodel):
#"""
	print("SVM binary Classification")
	start_date = "2013-01-01"
	end_date = "2017-01-01"
	dates = pd.date_range(start_date, end_date)	
	df_out = pd.DataFrame(index=dates)
	df_out["Real"] = y_test.copy()
	df_out=df_out.dropna()
	n_components=len(X_train.columns)-1
	pca = PCA(copy=True,n_components=n_components, whiten=False).fit(X_train)
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	#print(n_components)
 	#c = parameters[0]
	#g =  parameters[1]
	#{{'C': 3500, 'kernel': 'sigmoid', 'gamma': 0.05}
      #{'kernel': 'rbf', 'C': 0.1, 'gamma': 100.0}

	clf = SVC(C=3500,kernel= 'sigmoid', gamma= 0.05)

	start = datetime.datetime.now()
	clf.fit(X_train_pca, y_train)
	end = datetime.datetime.now()
	training=end-start
	accuracy = clf.score(X_test_pca, y_test)
	start = datetime.datetime.now()
	df_out['SVM'] = clf.predict(X_test_pca) 
	predictions = clf.predict(X_test_pca)
	end = datetime.datetime.now()
	Sscore = f1_score(y_test, predictions,average='binary')
	#cm = confusion_matrix(y_test, predictions)
	#plot_cm(cm,"Support Vector Machines")
	roc=mat=0
	if kFOLDS != 2:
         roc = roc_auc_score(y_test, predictions)
         mat = matthews_corrcoef(y_test, predictions) 
	predecir=end-start
	#write_Ex(df_out, "df_out.xlsx", "df_out") 
	#plotvalidation_curve(y_test,predictions)
	print("F1 Score SVM " +str(Sscore))
	return accuracy, Sscore, y_test,predictions, training, predecir,roc,mat

def performAdaBoostClass(X_train, y_train, X_test, y_test):#, parameters, fout, savemodel):

	print("Ada Boosting binary Classification")
    #{'algorithm': 'SAMME.R', 'n_estimators': 80, 'learning_rate': 0.7}
	clf = AdaBoostClassifier(n_estimators=80, algorithm='SAMME.R', learning_rate= 0.7)
	#clf = AdaBoostClassifier() 
	start = datetime.datetime.now()
	clf.fit(X_train, y_train)
	end = datetime.datetime.now()
	training=end-start

	accuracy = clf.score(X_test, y_test)
	start = datetime.datetime.now()
	predictions = clf.predict(X_test)
	end = datetime.datetime.now()
	Sscore = f1_score(y_test, predictions)
	roc=mat=0
	if kFOLDS != 2:
         roc = roc_auc_score(y_test, predictions)
         mat = matthews_corrcoef(y_test, predictions)   
	#cm = confusion_matrix(y_test, predictions)
	#plot_cm(cm,"Ada Boosting binary Classification") 
	predecir=end-start
	print("F1 Score ADA " +str(Sscore))
	
	return accuracy, Sscore, y_test,predictions, training, predecir,roc,mat

def performGTBClass(X_train, y_train, X_test, y_test):#, parameters, fout, savemodel):
    #"""
	print("Gradient Tree Boosting binary Classification")
 #{'max_features': 'auto', 'min_samples_split': 300, 'min_samples_leaf': 75, 'n_estimators': 90, 'learning_rate': 0.1}
	clf = GradientBoostingClassifier(min_samples_leaf=75, n_estimators=90, max_features= 'auto',min_samples_split=300, learning_rate=0.1)
	#clf = GradientBoostingClassifier()
	start = datetime.datetime.now()
	clf.fit(X_train, y_train)
	end = datetime.datetime.now()
	training=end-start

	accuracy = clf.score(X_test, y_test)
	start = datetime.datetime.now()
	predictions = clf.predict(X_test)
	end = datetime.datetime.now()
	Sscore = f1_score(y_test, predictions)
	roc=mat=0
	if kFOLDS != 2:
         roc = roc_auc_score(y_test, predictions)
         mat = matthews_corrcoef(y_test, predictions)  
	#cm = confusion_matrix(y_test, predictions)
	#plot_cm(cm,"Gradient Tree Boosting Classification") 
	predecir=end-start
	print("F1 Score GTB " +str(Sscore))
	return accuracy, Sscore, y_test,predictions, training, predecir,roc,mat
	
def performLRBClass(X_train, y_train, X_test, y_test):#, parameters, fout, savemodel):
	print("Logistic Regresion Binary Classification")

	clf = LogisticRegression(penalty= 'l1', C= 1, solver= 'liblinear') #{'penalty': 'l1', 'C': 1, 'solver': 'liblinear'}
	start = datetime.datetime.now()
	clf.fit(X_train, y_train)
	end = datetime.datetime.now()
	training=end-start
	accuracy = clf.score(X_test, y_test)
	start = datetime.datetime.now()
	predictions = clf.predict(X_test)
	end = datetime.datetime.now()
	Sscore = f1_score(y_test, predictions)
	roc=mat=0
	if kFOLDS != 2:
         roc = roc_auc_score(y_test, predictions)
         mat = matthews_corrcoef(y_test, predictions)  
	predecir=end-start
	#cm = confusion_matrix(y_test, predictions)
	#plot_cm(cm,"Gaussian Naives Bayes Classification") 
	print("F1 Score Logistic Regresion " +str(Sscore))
	return accuracy, Sscore, y_test,predictions, training, predecir,roc,mat
	
def performLDAClass(X_train, y_train, X_test, y_test):#, parameters, fout, savemodel):
	print("Linear Discriminant Analysis binary Classification")
#'{'store_covariance': 'True', 'n_components': 1, 'solver': 'lsqr', 'shrinkage': 0.1}
	clf = LinearDiscriminantAnalysis(store_covariance=True, n_components= 1, tol=0.5, solver='svd')
	#clf = LinearDiscriminantAnalysis() 
	start = datetime.datetime.now()
	clf.fit(X_train, y_train)
	end = datetime.datetime.now()
	training=end-start
	accuracy = clf.score(X_test, y_test)
	start = datetime.datetime.now()
	predictions = clf.predict(X_test)
	end = datetime.datetime.now()
	Sscore = f1_score(y_test, predictions)
	roc=mat=0
	if kFOLDS != 2:
         roc = roc_auc_score(y_test, predictions)
         mat = matthews_corrcoef(y_test, predictions)  
	predecir=end-start
	print("F1 Score LDA " +str(Sscore))
	#cm = confusion_matrix(y_test, predictions)
	#plot_cm(cm,"Linear Discriminant Analysis Classification")     
	return accuracy, Sscore, y_test,predictions, training, predecir,roc,mat

def performVotingClass(X_train, y_train, X_test, y_test):#, parameters, fout, savemodel):
	print("Voting binary Classification")
    
	#clf = LinearDiscriminantAnalysis()
	clf1 = SVC(C=2000,kernel= 'linear')
	clf2 = neighbors.KNeighborsClassifier(algorithm='ball_tree', n_neighbors=350, weights='distance', leaf_size= 30)  
	#clf2 = RandomForestClassifier(random_state=1)
	clf3 = LogisticRegression(penalty= 'l1', C= 1, solver= 'liblinear')   
	clf4 = SGDClassifier( alpha= 0.01,penalty='elasticnet', loss='perceptron', learning_rate= 'invscaling', eta0=0.2)
	eclf = VotingClassifier(estimators=[('svc', clf1), ('knn', clf2), ('lgr', clf3),('sdg',clf4)], voting='hard')

	start = datetime.datetime.now()     

	clf1.fit(X_train, y_train)
	clf2.fit(X_train, y_train)
	clf3.fit(X_train, y_train)
	clf4.fit(X_train, y_train)
	eclf.fit(X_train, y_train)
 
   
	end = datetime.datetime.now()
	training=end-start
	accuracy = eclf.score(X_test, y_test)
	start = datetime.datetime.now()
	predictions = eclf.predict(X_test)
	end = datetime.datetime.now()
	Sscore = f1_score(y_test, predictions)
	roc=mat=0
	if kFOLDS != 2:
         roc = roc_auc_score(y_test, predictions)
         mat = matthews_corrcoef(y_test, predictions) 
	predecir=end-start
	print("F1 Score Voting Classifier " +str(Sscore))
	#cm = confusion_matrix(y_test, predictions)
	#plot_cm(cm,"Voting binary Classification")       
	return accuracy, Sscore, y_test,predictions, training, predecir,roc,mat

def performDTCClass(X_train, y_train, X_test, y_test):#, parameters, fout, savemodel):
    #"""
	print("DecisionTreeClassifier Classification")
    #"""
	clf = DecisionTreeClassifier(max_features='sqrt', criterion='entropy') #{'max_features': 'sqrt', 'criterion': 'entropy'}

	start = datetime.datetime.now()
	clf.fit(X_train, y_train)
	end = datetime.datetime.now()
	training=end-start
	accuracy = clf.score(X_test, y_test)
	start = datetime.datetime.now()
	predictions = clf.predict(X_test)
	end = datetime.datetime.now()
	Sscore = f1_score(y_test, predictions)
	roc=mat=0
	if kFOLDS != 2:
         roc = roc_auc_score(y_test, predictions)
         mat = matthews_corrcoef(y_test, predictions) 
	predecir=end-start
	#cm = confusion_matrix(y_test, predictions)
	#plot_cm(cm,"Decision Tree Classification")    
	print("F1 Score DecisionTreeClassifier " +str(Sscore))
	return accuracy, Sscore, y_test,predictions, training, predecir,roc,mat	


def GSLDA(X_train, y_train, X_test, y_test,test):
#Choose all predictors except target & IDcols
	def f1_score_on_test(estimator, x, y):
		y_pred = estimator.predict(X_test)
		return f1_score(y_test.values, y_pred)
		
	params_map = [{'solver':['lsqr','eigen'],'shrinkage':['auto',0.1,0.3,0.5,0.7,0.9],'store_covariance':['True', 'False'],'n_components':[1,2,10,20,30,40,50,60, 80, 100,200, 300,500,1000]},{'solver':['svd'], 'tol':[0.3,0.45,0.5,0.55,0.8, 1.0],'store_covariance':['True', 'False'],'n_components':[1,2,10,20,30,40,50,60, 80, 100,200, 300,500,1000]}]
	clf = GridSearchCV(LinearDiscriminantAnalysis(), param_grid =  params_map, scoring=f1_score_on_test, verbose=100)
	clf.fit(X_train, y_train)
	df=pd.DataFrame(clf.grid_scores_)
	total_file = os.path.dirname(os.path.realpath(__file__)) + "\\CV\\CVLDA_" + str(test)+ ".xlsx"  
	writer = pd.ExcelWriter(total_file)
	df.to_excel(writer,"LDA")
	writer.save() 			
	
	return clf.best_params_

	
	
def GSGTB(X_train, y_train, X_test, y_test,test):
#Choose all predictors except target & IDcols
	def f1_score_on_test(estimator, x, y):
		y_pred = estimator.predict(X_test)
		return f1_score(y_test.values, y_pred)
		
	param_test1 = {'n_estimators':[20,40,50,60,70,80,90,100,150,200],'learning_rate':[0.1,0.2], 'min_samples_split':[100,200,300],'min_samples_leaf':[10,30,40, 50, 75],'max_features':['auto', 'sqrt']} 
     #,'subsample':[0.6,0.8,1.0],'random_state':[10,20,30]}
	#param_test2 = [{'n_estimators':[20,40,50,60,70,80,90,100,150,200]}]
	clf = GridSearchCV(GradientBoostingClassifier(), param_grid = param_test1, scoring=f1_score_on_test,n_jobs=1,iid=False, cv=5, verbose=100)
	#clf = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10),param_grid = param_test2, scoring='roc_auc',n_jobs=1,iid=False, cv=5, verbose=100)
	clf.fit(X_train, y_train)
	df=pd.DataFrame(clf.grid_scores_)
	total_file = os.path.dirname(os.path.realpath(__file__)) + "\\CV\\CVGTB_" + str(test)+ ".xlsx"  
	writer = pd.ExcelWriter(total_file)
	df.to_excel(writer,"GTB")
	writer.save() 			
	return clf.best_params_


def GSSGD(X_train, y_train, X_test, y_test,test):
#Choose all predictors except target & IDcols
	def f1_score_on_test(estimator, x, y):
		y_pred = estimator.predict(X_test)
		return f1_score(y_test.values, y_pred)

	print("Stochastic Gradient Descent binary Classification")

	#c = parameters[0]
	#parameters = {'loss': [ 'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001], 'n_iter': list(np.arange(1,1001))}
	#clf = SVC(C=5, kernel= 'rbf', gamma= 0.01) {'kernel': 'rbf', 'C': 100.0, 'gamma': 0.5}
	param_test1=[{'loss':['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
 	              'penalty':['none', 'l2', 'l1', 'elasticnet'],
 	              'learning_rate':['constant','optimal','invscaling'],
	              'eta0':[0.1,0.2], 'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}]
	clf = GridSearchCV(SGDClassifier(), param_grid = param_test1, scoring=f1_score_on_test,n_jobs=1,iid=False, cv=5, verbose=100)
	#clf = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10),param_grid = param_test2, scoring='roc_auc',n_jobs=1,iid=False, cv=5, verbose=100)
	clf.fit(X_train, y_train)
	df=pd.DataFrame(clf.grid_scores_)
	total_file = os.path.dirname(os.path.realpath(__file__)) + "\\CV\\CVSGD_" + str(test)+ ".xlsx"  
	writer = pd.ExcelWriter(total_file)
	df.to_excel(writer,"SGD")
	writer.save() 			
	return clf.best_params_





def GSADA(X_train, y_train, X_test, y_test,test):	
	def f1_score_on_test(estimator, x, y):
		y_pred = estimator.predict(X_test)
		return f1_score(y_test.values, y_pred)#return f1_score(y_test.values, y_pred, pos_label=1)


	param_grid = [{'n_estimators':[20,40,50,60,70,80,90,100,150,200],'algorithm' : ['SAMME', 'SAMME.R'], 'learning_rate':[0.1, 0.3,0.5,0.7,1] }]


	DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced")
	#DTC = SVC(kernel= 'rbf', C= 10.0, gamma= 0.1)
	ABC = AdaBoostClassifier(base_estimator = DTC)


	clf = GridSearchCV(ABC, param_grid=param_grid, scoring=f1_score_on_test, verbose=100 )
	clf.fit(X_train, y_train)
	df=pd.DataFrame(clf.grid_scores_)
	total_file = os.path.dirname(os.path.realpath(__file__)) + "\\CV\\CVADA_" + str(test)+ ".xlsx"  
	writer = pd.ExcelWriter(total_file)
	df.to_excel(writer,"ADA")
	writer.save() 			
 
	return clf.best_params_

def GSDTC(X_train, y_train, X_test, y_test,test):	
	def f1_score_on_test(estimator, x, y):
		y_pred = estimator.predict(X_test)
		return f1_score(y_test.values, y_pred)


	param_grid = [{'criterion':['gini','entropy'],'max_features':['auto','sqrt','log2' ] }]


	clf = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, scoring=f1_score_on_test, verbose=100 )
	clf.fit(X_train, y_train)
	df=pd.DataFrame(clf.grid_scores_)
	total_file = os.path.dirname(os.path.realpath(__file__)) + "\\CV\\CVDTC_" + str(test)+ ".xlsx"  
	writer = pd.ExcelWriter(total_file)
	df.to_excel(writer,"DTC")
	writer.save() 			
 
	return clf.best_params_

	
def GSSVC(X_train, y_train, X_test, y_test, test):

    def f1_score_on_test(estimator, x, y):
        y_pred = estimator.predict(X_test)
        return f1_score(y_test.values, y_pred)
# TODO: Create the parameters list you wish to tune
    params_map = [{ 'C': [0.001,0.005, 0.01,0.02, 0.1,0.4, 0.5,0.6,0.7, 1.0, 2.0, 5.0, 10.0, 100.0],'kernel': ['rbf'],'gamma': ['auto',0.001,0.01, 0.05,0.1, 0.5, 1.0,4.0, 5.0, 6.0, 10.0,50.0,100.0,150.0]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000, 1500, 2000, 2500,3000, 3500,4000]}]#,
    #{'kernel': ['poly'], 'C': [1, 10, 100, 1000, 1500, 2000, 2500,3000, 3500,4000],'degree':[2,3,4,5,6,7,8,9,10,15], 'gamma': ['auto',0.001,0.01, 0.05,0.1, 0.5, 1.0, 5.0]},
    #{'kernel': ['sigmoid'], 'C': [1, 10, 100, 1000, 1500, 2000, 2500,3000, 3500,4000], 'gamma': ['auto',0.001,0.01, 0.05,0.1, 0.5, 1.0, 5.0]}]


    clf = GridSearchCV(SVC(), params_map, scoring=f1_score_on_test, verbose=100)



    clf.fit(X_train, y_train)
    #joblib.dump(clf.best_estimator_, file_path, compress = 3)
    df=pd.DataFrame(clf.grid_scores_)
    total_file = os.path.dirname(os.path.realpath(__file__)) + "\\CV\\CVSVC_" + str(test)+ ".xlsx" 
    writer = pd.ExcelWriter(total_file)
    df.to_excel(writer,"SVC")
    writer.save()

    return clf.best_params_



def GSKNN(X_train, y_train, X_test, y_test, test):
    
    def f1_score_on_test(estimator, x, y):
        y_pred = estimator.predict(X_test)
        return f1_score(y_test.values, y_pred)

# here must be some code for your training set and test set


    tuned_parameters = [{'n_neighbors': [50,100,150,200,250,300,350], 'weights': ['distance', 'uniform'],'algorithm': ['ball_tree', 'kd_tree', 'brute'], 'leaf_size':[30,40,50,60] }]


    clf = GridSearchCV(neighbors.KNeighborsClassifier(), tuned_parameters, cv=10, scoring=f1_score_on_test, verbose=100)
    clf.fit(X_train, y_train)
    df=pd.DataFrame(clf.grid_scores_)
    total_file = os.path.dirname(os.path.realpath(__file__)) + "\\CV\\CVKNN_" + str(test)+ ".xlsx"  
    writer = pd.ExcelWriter(total_file)
    df.to_excel(writer,"KNN")
    writer.save()
    return clf.best_estimator_


 
def GSRFC(X_train, y_train, X_test, y_test,test):
    def f1_score_on_test(estimator, x, y):
        y_pred = estimator.predict(X_test)
        return f1_score(y_test.values, y_pred)

# here must be some code for your training set and test set


    tuned_parameters = [{'n_estimators': [500,1000,2000,2500,3000,3500,4000, 4500,5000],
    'criterion':['gini','entropy'] }]

    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, scoring=f1_score_on_test, verbose=100)
    clf.fit(X_train, y_train)
    df=pd.DataFrame(clf.grid_scores_)
    total_file = os.path.dirname(os.path.realpath(__file__)) + "\\CV\\CVRFC_" + str(test)+ ".xlsx" 
    writer = pd.ExcelWriter(total_file)
    df.to_excel(writer,"RFC")
    writer.save()
    return clf.best_params_

def GSLRC(X_train, y_train, X_test, y_test,test):
    def f1_score_on_test(estimator, x, y):
        y_pred = estimator.predict(X_test)
        return f1_score(y_test.values, y_pred)


    param_grid = [{'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty' :["l1"], 'solver' :["liblinear"] },
                  {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty' :["l2"], 'solver' :[ "newton-cg", "lbfgs", "sag"] }]

    clf = GridSearchCV(LogisticRegression(), param_grid, scoring=f1_score_on_test, verbose=100)
    clf.fit(X_train, y_train)
    df=pd.DataFrame(clf.grid_scores_)
    total_file = os.path.dirname(os.path.realpath(__file__)) + "\\CV\\CVLRC_" + str(test)+ ".xlsx"  
    writer = pd.ExcelWriter(total_file)
    df.to_excel(writer,"LRC")
    writer.save()
    return clf.best_params_

	
def TimeSeriesCrossValidation(X_train, y_train, number_folds, algorithm):#	algorithm=["KNN","RFC","SVM","ADA Bost","GTB","LDA"]
	
	#print('Parameters --------------------------------> ', parameters)
	print('Size train set: '+ str(X_train.shape))
    

	k = int(np.floor(float(X_train.shape[0]) / number_folds))
	print('Size of each fold: '+ str(k))
    

	accuracies = np.zeros(number_folds-1)
	F1scores = np.zeros(number_folds-1)#
	rocs= np.zeros(number_folds-1)
	mats= np.zeros(number_folds-1)

    # loop from the first 2 folds to the total number of folds    
	for i in range(2, number_folds + 1):
		print('')

		split = float(i-1)/i

		print('Splitting the first ' + str(i) + ' chunks at ' + str(i-1) + '/' + str(i))
		X = X_train[:(k*i)]
		y = y_train[:(k*i)]

        # split percentage we have set above
		index = int(np.floor(X.shape[0] * split))
        
        # folds used to train the model        
		X_trainFolds = X[:index]        
		y_trainFolds = y[:index]
        
        # fold used to test the model
		X_testFolds = X[(index + 1):]
		y_testFolds = y[(index + 1):]

        #	algorithm=["KNN","RFC","SVM","ADA Bost","GTB","LDA"]
		if algorithm=="ADA Bost":
			accuracies[i-2], F1scores[i-2], y_test_,predict, training, predecir,rocs[i-2], mats[i-2] = performAdaBoostClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="LDA":
			accuracies[i-2], F1scores[i-2], y_test_,predict, training, predecir,rocs[i-2], mats[i-2] = performLDAClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="GTB":
			accuracies[i-2], F1scores[i-2], y_test_,predict, training, predecir,rocs[i-2], mats[i-2] = performGTBClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="KNN":
			accuracies[i-2], F1scores[i-2], y_test_,predict, training, predecir,rocs[i-2], mats[i-2] = performKNNClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="RFC":
			accuracies[i-2], F1scores[i-2], y_test_,predict, training, predecir,rocs[i-2], mats[i-2] =  performRFClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="SVM":
			accuracies[i-2], F1scores[i-2], y_test_,predict, training, predecir,rocs[i-2], mats[i-2] = performSVMClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="SGD":
			accuracies[i-2], F1scores[i-2], y_test_,predict, training, predecir,rocs[i-2], mats[i-2] = performSGDClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="VOT":
			accuracies[i-2], F1scores[i-2], y_test_,predict, training, predecir,rocs[i-2], mats[i-2] = performVotingClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="LRC":
			accuracies[i-2], F1scores[i-2], y_test_,predict, training, predecir,rocs[i-2], mats[i-2] = performLRBClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="DTC":
			accuracies[i-2], F1scores[i-2], y_test_,predict, training, predecir,rocs[i-2], mats[i-2] = performDTCClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
       

		print('Accuracy on fold ' + str(i) + ': ' + str(accuracies[i-2]))
		print('F1 Score on fold ' + str(i) + ': ' + str(F1scores[i-2]))
		print('Roc on fold ' + str(i) + ': ' + str(rocs[i-2]))
		print('Matthew Coef on fold ' + str(i) + ': ' + str(mats[i-2]))
 
	return accuracies.mean(), F1scores.mean(), y_test_,predict, training, predecir,rocs.mean(),mats.mean()

def call_alg(X_train, y_train, X_test, y_test, alg):#	
    
    if alg=="KNN":    
        score, f1score, y_test,predictions, training, predecir,roc,mat = performKNNClass(X_train, y_train, X_test, y_test)
    if alg=="RFC":   
        score, f1score, y_test,predictions, training, predecir,roc,mat = performRFClass(X_train, y_train, X_test, y_test)
    if alg=="SVM": 
        score, f1score, y_test,predictions, training, predecir,roc,mat = performSVMClass(X_train, y_train, X_test, y_test)
    if alg=="ADA Bost": 
        score, f1score, y_test,predictions, training, predecir,roc,mat = performAdaBoostClass(X_train, y_train, X_test, y_test)
    if alg=="GTB": 
        score, f1score, y_test,predictions, training, predecir,roc,mat = performGTBClass(X_train, y_train, X_test, y_test)
    if alg=="LDA": 
        score, f1score, y_test,predictions, training, predecir,roc,mat = performLDAClass(X_train, y_train, X_test, y_test)
    if alg=="SGD": 
        score, f1score, y_test,predictions, training, predecir,roc,mat = performSGDClass(X_train, y_train, X_test, y_test)
    if alg=="VOT": 
        score, f1score, y_test,predictions, training, predecir,roc,mat = performVotingClass(X_train, y_train, X_test, y_test)
    if alg=="LRC": 
        score, f1score, y_test,predictions, training, predecir,roc,mat = performLRBClass(X_train, y_train, X_test, y_test)
    if alg=="DTC": 
        score, f1score, y_test,predictions, training, predecir,roc,mat = performDTCClass(X_train, y_train, X_test, y_test)


        
    return score, f1score, y_test,predictions, training, predecir,roc,mat

def classifaction_report_csv(report,alg,version,TEST,kFOLDS, score, training, predecir,roc,mat):
    path = os.path.dirname(os.path.realpath(__file__))
    ahora=str(datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S"))
		
    file_path = path +  "\\out\\" + "classification_report_"+str(alg) + ahora+  ".csv"
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['Algorithm'] = str(alg)
        row['Accuracy'] = str(score)
        row['roc_auc_score'] = str(roc)
        row['matthews_corrcoef'] = str(mat)        
        row['Features'] = str(version)
        row['Version'] = str(TEST)
        row['KFolds'] = str(kFOLDS)
        row['Date'] = str(ahora)
        row['Training'] = str(training)
        row['Predicting'] = str(predecir)
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(file_path, index = False)
    
def Walk_Forward_Validation_CV(df, start_test, algorithm):#	algorithm=["KNN","RFC","SVM","ADA Bost","GTB","LDA"]
#def prepareDataForClassification(dataset, start_test):

	le = preprocessing.LabelEncoder()
    
	df['UpDown'] = df.apply (lambda row: Label_Change (row),axis=1)
	dataset1=df.truncate(before='2003-07-01') #delete all values up to the first rolling average 63 days
	dataset1.UpDown = le.fit(dataset1.UpDown).transform(dataset1.UpDown)
    
	features = dataset1.columns[1:-1]
	X = dataset1[features]    
	y = dataset1.UpDown    
    
	X_train = X[X.index < start_test]
	y_train = y[y.index < start_test]              
    

    
	n_train = len(X_train)
	n_records= len(dataset1)
	number_folds=n_records-n_train
	accuracies = np.zeros(number_folds-1)
	F1scores = np.zeros(number_folds-1)
	rocs= np.zeros(number_folds-1)
	mats= np.zeros(number_folds-1) 
	yreals = np.zeros(number_folds-1)
	ypredictions = np.zeros(number_folds-1)
    # loop from the first 2 folds to the total number of folds    
	for i in range(n_train, n_records-1):

   
		X_trainFolds = X[0:i]        
		y_trainFolds = y[0:i]

        # fold used to test the model
		X_testFolds = X[i:i+1]
		y_testFolds = y[i:i+1]
 
       #	algorithm=["KNN","RFC","SVM","ADA Bost","GTB","LDA", "SGD","GNB", "VOT", "DTC"]
		if algorithm=="ADA Bost":
			accuracies[i-n_train], F1scores[i-n_train],yreals[i-n_train], ypredictions[i-n_train], training, predecir,rocs[i-n_train], mats[i-n_train]  = performAdaBoostClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="LDA":
			accuracies[i-n_train], F1scores[i-n_train],yreals[i-n_train], ypredictions[i-n_train], training, predecir,rocs[i-n_train], mats[i-n_train]  = performLDAClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="GTB":
			accuracies[i-n_train], F1scores[i-n_train],yreals[i-n_train], ypredictions[i-n_train], training, predecir,rocs[i-n_train], mats[i-n_train]  = performGTBClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="KNN":
			accuracies[i-n_train], F1scores[i-n_train],yreals[i-n_train], ypredictions[i-n_train], training, predecir,rocs[i-n_train], mats[i-n_train]  = performKNNClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="RFC":
			accuracies[i-n_train], F1scores[i-n_train],yreals[i-n_train], ypredictions[i-n_train], training, predecir,rocs[i-n_train], mats[i-n_train]  =  performRFClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="SVM":
			accuracies[i-n_train], F1scores[i-n_train],yreals[i-n_train], ypredictions[i-n_train], training, predecir,rocs[i-n_train], mats[i-n_train]  = performSVMClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="SGD":
			accuracies[i-n_train], F1scores[i-n_train],yreals[i-n_train], ypredictions[i-n_train], training, predecir,rocs[i-n_train], mats[i-n_train]  = performSGDClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="VOT":
			accuracies[i-n_train], F1scores[i-n_train],yreals[i-n_train], ypredictions[i-n_train], training, predecir,rocs[i-n_train], mats[i-n_train]  = performVotingClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="LRC":
			accuracies[i-n_train], F1scores[i-n_train],yreals[i-n_train], ypredictions[i-n_train], training, predecir,rocs[i-n_train], mats[i-n_train]  = performLRBClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
		if algorithm=="DTC":
			accuracies[i-n_train], F1scores[i-n_train],yreals[i-n_train], ypredictions[i-n_train], training, predecir,rocs[i-n_train], mats[i-n_train]  = performDTCClass(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds)
      

		print('Accuracy on fold ' + str(i) + ': ' + str(accuracies[i-n_train]))
		print('F1 Score on fold ' + str(i) + ': ' + str(F1scores[i-n_train]))
		print('ytest fold ' + str(i) + ': ' + str(yreals[i-n_train]))
		print('Prediction on fold ' + str(i) + ': ' + str(ypredictions[i-n_train]))
		print('Roc on fold ' + str(i) + ': ' + str(rocs[i-n_train]))
		print('Matthew Coef on fold ' + str(i) + ': ' + str(mats[i-n_train]))
  
	return accuracies.mean(), F1scores.mean(),yreals, ypredictions,training, predecir,rocs.mean(),mats.mean()

# Plot CV scores of a 2D grid search
def plotGridResults2D(x, y, x_label, y_label, grid_scores):

    scores = [s[1] for s in grid_scores]
    scores = np.array(scores).reshape(len(x), len(y))

    plt.figure()
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.RdYlGn)
    plt.xlabel(y_label)
    plt.ylabel(x_label)
    plt.colorbar()
    plt.xticks(np.arange(len(y)), y, rotation=45)
    plt.yticks(np.arange(len(x)), x)
    plt.title('Validation accuracy')