import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import plotly.express as px

dataset1 = pd.read_csv("C:\\Users\\HP\\Desktop\\java2\\bitcoin_Price_Movement_Clustering.csv")

dataset1['Date'] = pd.to_datetime(dataset1['Date'], format='%d-%m-%Y')

print(dataset1.head(5))

print(dataset1.shape)

print(dataset1.describe())

print(dataset1.info())

numeric_columns = dataset1.select_dtypes(include=[np.number]).columns
correlation_matrix = dataset1[numeric_columns].corr()

print(correlation_matrix)

dataset1['Price Movement'] = dataset1['Price Movement'].apply(lambda x: 1 if x == 'Increase' else (0 if x == 'Decrease' else x))

if 'converted_column' in dataset1.columns:
        dataset1.drop('converted_column', axis=1, inplace=True)

output_file_path = 'output_file1.csv'
dataset1.to_csv(output_file_path, index=False)

import os
if os.path.exists(output_file_path):
    print(f"File '{output_file_path}' successfully created.")
else:
    print(f"Error: File '{output_file_path}' could not be created.")
dataset1.drop('converted_column', axis=1, inplace=False)
dataset1.head(4)
x=dataset1.iloc[:,[1,4]].values
y=dataset1.iloc[:,-1].values
x
y
x_train,x_test,y_train,ytest = train_test_split(x,y,test_size=0.25)
sc=Standardscaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
cla=RandomForestClassifier()
cla.fit(x_train,y_train)
pred=cla.predict(x_test)
pred
for x in range(len(x_test)):
  print(y_test[x],pred[x])
confusion_matrix(y_test,pred)
accuracy_score(y_test,pred)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
x_set,y_set = sc.inverse_transform(x_train),y_train
x1,x2 = np.meshgrid(np.arange(start=x_set[:,0].min()-10,stop=x_set[:,0].max()+10, step=1),np.arange(start=x_set[:,1].min()-1000,stop=x_set[:,1].max()+1000,step = 1))
plt.contourf(x1,x2,cla.predict(sc.transform(np.array([x1.ravel(),x2.ravel()]).T)).reshape(x1.shape),
alpha=0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
  plt.scatter(x_set[y_set == j,0],x_set[y_set == j,1],c = ListedColormap(('red','green'))(i),label=j)
plt.title('Random_Forest')
plt.legend()
plt.show()
clus=SVC(kernel='sigmoid')
clus.fit(x_train,y_train)
predic=clus.predict(x_test)
predic
confusion_matrix(y_test,predic)
accuracy_score(y_test,predic)
for x in range(len(x_test)):
  print(y_test[x],predic[x])
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
x_set,y_set = sc.inverse_transform(x_train),y_train
x1,x2 = np.meshgrid(np.arange(start=x_set[:,0].min()-10,stop=x_set[:,0].max()+10, step=1),np.arange(start=x_set[:,1].min()-1000,stop=x_set[:,1].max()+1000,step = 1))
plt.contourf(x1,x2,clus.predict(sc.transform(np.array([x1.ravel(),x2.ravel()]).T)).reshape(x1.shape),
alpha=0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
  plt.scatter(x_set[y_set == j,0],x_set[y_set == j,1],c = ListedColormap(('red','green'))(i),label=j)
plt.title('SVM Model')
plt.legend()
plt.show()
import plotly.express as px
fig = px.scatter_3d(dataset1,x='Low',y='High',z='Volume',color='High',
                    size='Low',title="size--> Low    color--> High" )
fig.show()
dataset1.head(5)
import seaborn as sns
sns.set_context("talk",font_scale=1.3)
with sns.axes_style("darkgrid"):
  fig,ax=plt.subplots(figsize=(16,8))
  sns.lineplot(x=dataset1.Date,y=dataset1.Close,color='purple')
  ax.set_title('Daily Closing Price')
  ax.set_xlabel('Date: From Jan. 1, 2019 to Dec. 31, 2021')