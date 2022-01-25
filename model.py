import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("cardio_train.csv", sep=";")

print(df.head())

df.drop('id', axis = 1, inplace = True)
df.info()

dp = df[df.duplicated(keep=False)]
dp = dp.sort_values(by=['age', "gender", "height"], ascending= False)
dp.head(2)

df.drop_duplicates(inplace= True)
print("{} rows are same".format(df.duplicated().sum()))

df["bmi"] = (df["weight"] / (df["height"] / 100)**2).round(1)
df.head()
    
df = df[(df["bmi"]>10) & (df["bmi"]<100)]

df.drop(["weight","height"],axis = 1,inplace = True)

df = df[(df['ap_hi'] < 250) & (df['ap_lo'] < 200)]
df = df[(df['ap_hi'] > 20) & (df['ap_lo'] > 20)]

df['age'] =  df['age'] / 365

print(df.tail())

df['cholesterol'].unique()

df['cholesterol'] = df['cholesterol'].map({ 1: 'normal', 2: 'aboveNormal', 3: 'wellAboveNormal'})
df['gluc']=df['gluc'].map({ 1: 'normal', 2: 'aboveNormal', 3: 'wellAboveNormal'})

dummies = pd.get_dummies(df[['cholesterol','gluc']])


final_df = pd.concat([df,dummies],axis=1)
final_df.drop(['cholesterol','gluc'],axis=1,inplace=True)
final_df.head()

final_df["gender"] = final_df["gender"] % 2
print(final_df.tail())

y = final_df["cardio"]
X = final_df.drop("cardio", axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

logreg = LogisticRegression(solver='liblinear')

logreg.fit(X_train,y_train)

y_pred = pd.Series(logreg.predict(X_test))



print(accuracy_score(y_test,y_pred))

import pickle

pickle.dump(logreg,open('model.pkl','wb'))
