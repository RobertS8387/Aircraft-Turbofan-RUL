#|**********************************************************************;
#* Project           : Comparative Analysis on Machine Learning Algorithms for
#                      Predictive Maintenance in the Aviation Industry 
#*
#* Program name      : RandomForestRegression.py
#*
#* Author            : Robert Sutherland
#*
#* Date created      : 22/04/20
#*
#* Purpose           : To predict the RUL of an aircraft turbofan engine after
#                      a HPC fault occurs
#*
#* 
#* Extra Information : The code used is an improvement and a variation of Rutgher
#*                     Rhigart's code for the RUL of an aircraft turbofan engine
#*                     this can be accessed here: https://github.com/RRighart/Gatu/blob/master/Gatu-script.ipynb
#*
#*
#* Date        Author                
#* 22/04/20    Robert Sutherland 
#*
#|**********************************************************************;


#import libraries
from tabulate import tabulate 
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

#Function used to express time to failure as a fraction
def fractionTTF(dat,q):
    #(time to failure - min. time to failure)/(max. time to failure - min. time to failure) 
    return(dat.TTF[q]-dat.TTF.min()) / float(dat.TTF.max()-dat.TTF.min())
    
    
#Function for estimating the predicted total number of cycles per unit 
def totcycles(data):
    #cycles /(1 - predicted fraction TTF)
    return(data['cycles'] / (1-data['score']))

#Function to calculate RUL
def RULfunction(data):
    #(max. predicted cycles) - (max. cycles)
    return(data['maxpredcycles'] - data['maxcycles'])


#The following settings are used to avoid exponential values in output or 
#tables and to display 50 rows maximum
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', 10) 
pd.options.display.max_rows=50


#reads a training dataset, a test dataset, and RUL dataset (this holds the 
#true values for RUL) 
train = pd.read_csv('Train.csv', parse_dates=False, decimal=".", header=None)
test = pd.read_csv('Test.csv', parse_dates=False, decimal=".", header=None)
RUL = pd.read_csv('RUL.csv', parse_dates=False, decimal=".", header=None)

#removes columns with standard deviation >0.001
train = train.loc[:, train.std() > 0.001]
test = test.loc[:, test.std() > 0.001]


#add column names
train.columns = ['unit', 'cycles', 'op_setting1', 's2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
test.columns = ['unit', 'cycles', 'op_setting1', 's2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']


#using the groupby and merge function the max. number of cycles for each engine
#are added as an extra column and renamed 'maxcycles'
train = pd.merge(train, train.groupby('unit', as_index=False)['cycles'].max(), how='left', on='unit')
train.rename(columns={"cycles_x": "cycles", "cycles_y": "maxcycles"}, inplace=True)


#This line calculates the time to failure for every row 
#(max. no. of cycles for that particular engine - no. of cycles for the engine)
train['TTF'] = train['maxcycles'] - train['cycles']

print (train['TTF'])



#Scaling
scaler = MinMaxScaler()
print (train.describe().transpose())

#Make a copy of the data so that there is a dataset for unscaled and scaled data
ntrain = train.copy()
ntest = test.copy()
#Select data needed for scaling
ntrain.iloc[:,2:18] = scaler.fit_transform(ntrain.iloc[:,2:18])
ntest.iloc[:,2:18] = scaler.transform(ntest.iloc[:,2:18])

print (ntrain.describe().transpose())


print (pd.DataFrame(ntest.columns).transpose())
print (ntest.describe().transpose())


#create empty arrays
fTTFz = []
fTTF = []

#iterate some integer i from min. value in engine to max. value in engine
for i in range(train['unit'].min(),train['unit'].max()+1):
    
    dat=train[train.unit==i]
    dat = dat.reset_index(drop=True)
    for q in range(len(dat)):
        fTTFz = fractionTTF(dat, q)
        fTTF.append(fTTFz)
ntrain['fTTF'] = fTTF

#assign training variables (sensor readings)
X_train = ntrain.values[:,1:18]
#assign true RUL values
Y_train = ntrain.values[:, 20]
#assign testing variables (sensor readings)
X_test = ntest.values[:,1:18]


#train the model
model = RandomForestRegressor(n_estimators=100, 
                               bootstrap = True,
                               random_state = 0) 


#fit model to training dataset
model.fit(X_train, Y_train)

#predict fraction RUL values for testset
score = model.predict(X_test)


#print (score[0:10])

#print(score.min(), score.max())


#re-convert fraction RUL to RUL cycles using similar method as before
test = pd.merge(test, test.groupby('unit', as_index=False)['cycles'].max(), how='left', on='unit')
test.rename(columns={"cycles_x": "cycles", "cycles_y": "maxcycles"}, inplace=True)
test['score'] = score

#print (test.head())

#call totcycles function to calculate the max. predicted cycles per unit in the testset
#adding a column named 'maxpredcycles' 
test['maxpredcycles'] = totcycles(test)

#call RUL function to obtain RUL 
#adding a column named 'RUL'
#this allows the RUL to be calculated at the point of the max. cycle reached in
#the test set 
test['RUL'] = RULfunction(test)


#the following will compute the RUL per unit (based on the max. cycles) 
#from the RUL column that contains predicted values for each
t = test.columns == 'RUL'
ind = [i for i, x in enumerate(t) if x]

predictedRUL = []

for i in range(test.unit.min(), test.unit.max()+1):
    npredictedRUL=test[test.unit==i].iloc[test[test.unit==i].cycles.max()-1,ind]
    predictedRUL.append(npredictedRUL)
    


xtrueRUL = list(RUL.loc[:,0])
otrueRUL = []

for i in range(0,len(xtrueRUL)):
    otrueRUL = np.concatenate((otrueRUL, list(reversed(np.arange(xtrueRUL[i])))))
    
xpredictedRUL = list(round(x) for x in predictedRUL)
opredictedRUL = []



new_xpredictedRUL = []

for i in range (0,len(xpredictedRUL)):
    test = xpredictedRUL[i].item()
    new_xpredictedRUL.append(test)


for i in range(0,len(xpredictedRUL)):
    testing = np.arange(xpredictedRUL[i].item())
    testing = reversed(testing)
    opredictedRUL = np.concatenate((opredictedRUL, list(testing)))
    

#identifies any differences in the DataFrame  by concatenating true and predicted
df1 = pd.concat([pd.Series(RUL[0]), pd.Series(new_xpredictedRUL)], axis=1)
df1.columns = ['true', 'predicted']

df1.index = np.arange(1, len(df1)+1)

df1 = df1.head(10)





#plot bar graph of predicted RUL vs True RUL
df1.plot(kind='bar',figsize=(30,24))
plt.title('Random Forest')
plt.xlabel('Engine ID')
plt.ylabel('Remaining Useful Life (No. of Cycles)')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#calculates the residual of predicted and true RUL 
df1['diff'] = df1['predicted']-df1['true']

#print True, Predicted, and Residual Values in table format
headers = ['True', 'Predicted', 'Residuals']
print('\n')
print(tabulate(df1.head(10), headers = headers))


#calculates overestimations and underestimations
est = pd.DataFrame({'Count': [(df1['diff']<0).sum(), (df1['diff']==0).sum(), (df1['diff']>0).sum()]}, columns=['Count'], index=['Smaller', 'Zero', 'Larger'])

#print (est)

#calculate MAE and RSME evaluation metrics
mae = mean_absolute_error(RUL, new_xpredictedRUL)
mse = mean_squared_error(RUL, new_xpredictedRUL)
rmse = math.sqrt(mse)
rmse = round(rmse, 2)
mae = round(mae, 2)
print('\n')
print(f'Root Mean Squared Error: {rmse}')
print('\n')
print(f'Mean Absolute Error: {mae}')
print('\n')
