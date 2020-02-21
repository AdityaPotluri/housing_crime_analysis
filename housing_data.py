"""
The following project uses a .txt file with information on boston housing data and several factors
mentioned below. The goal is to find the strongest correlation between a given housing data point
and the amount of crime in that neighborhood. The project uses sklearn's statistical tools to dete-
rmine the strongest correlation then graphs it with matplotlib. The Categories are as follows.

 CRIM     per capita crime rate by town
 ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
 INDUS    proportion of non-retail business acres per town
 CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 NOX      nitric oxides concentration (parts per 10 million)
 RM       average number of rooms per dwelling
 AGE      proportion of owner-occupied units built prior to 1940
 DIS      weighted distances to five Boston employment centres
 RAD      index of accessibility to radial highways
 TAX      full-value property-tax rate per $10,000
 PTRATIO  pupil-teacher ratio by town
 B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 LSTAT    % lower status of the population
 MEDV     Median value of owner-occupied homes in $1000's




"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

#cleans and formats data set removing any unecessary spaces or new line characters
def clean(a_string):
    a_string=a_string.replace("\n"," ")
    a_string=a_string.replace("   ","  ")
    a_string=a_string.replace("  ",",")
    a_string=a_string=list(a_string)
    for i in range(len(a_string)):
        if a_string[i]==" " and a_string[i+1]!=" ":
            a_string[i]="  "
    return ('').join(a_string)

#creates labels for our pandas dataframe
def create_labels():
    rows=[hous_stat[x:x+14]
          for x in range(0, len(hous_stat), 14)]
    return ["row "+str(rows.index(i)) for i in rows]

#creates rows for our pandas dataframe
def create_row(hous_stat,i):
    return [(np.array(hous_stat[x:x+i],dtype=np.float32))
            for x in range(0, len(hous_stat), i)]

#calculates the mean of a given category ex:LSTAT
def mean(category):
    list_sum=0
    for numbers in category:
       list_sum+=numbers
    return list_sum/len(category)

#generates and formats data so it is ready to be placed in a dictionary
def data_to_dict():
    row=create_row(hous_stat,14)
    categories=create_labels()
    return dict(zip(categories,row))






housing=open("housing.txt","r")
housing_r=clean(housing.read())



hous_stat=housing_r.split(",")
hous_stat=[i.replace(" ","") for i in hous_stat]
hous_df=pd.DataFrame(data_to_dict())
hous_df.rename(index={0:'CRIM',1:'ZN',2:'INDUS',3:'CHAS',4:'NOX',5:'RM',
              6:'AGE',7:'DIS',8:'RAD',9:'TAX',10:'PTRATIO',11:'B',
              12:'LSTAT',13:'MEDV'}, inplace=True)

a=hous_df.loc['CRIM']
b=(hous_df.drop(['CRIM']))



#creates statistical tools and analyzes them
y=np.array(a)
X=np.array(b)
X_train,X_test,y_train,y_test=train_test_split(X.reshape(506,13),y,test_size=0.1,random_state=10)
reg=KNeighborsRegressor(n_neighbors=2)
clf =reg.fit(X_train, y_train)
predicted_values=(clf.predict((X_test)))
real_values=(y_test)


print("mean squared error")
print(np.square(np.subtract(predicted_values, real_values)).mean())

#charts the strongest correlation with Crime in boston
x=hous_df.loc['MEDV']
y=hous_df.loc['CRIM']
plt.scatter(x,y)
plt.xlabel('Rooms Per Dwelling',fontsize=16)
plt.ylabel('CRIME',fontsize=16)
plt.show()
housing.close()
