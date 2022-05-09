import pandas as pd # library for data analysis
import re
import requests # library to handle requests
from bs4 import BeautifulSoup # library to parse HTML documents
import numpy as np
import matplotlib.pyplot as plt


################################################################
########  following code is for downloading and storing the data
#         in local disk
#         Uncomment the below snippet for first time running the 
#         code  :) 
"""
# get the response in the form of html
wikiurl="https://en.wikipedia.org/wiki/Transistor_count"
table_class="wikitable sortable jquery-tablesorter"
response=requests.get(wikiurl)
print(response.status_code)

# parse data from the html into a beautifulsoup object
soup = BeautifulSoup(response.text, 'html.parser')
indiatable=soup.find('table',{'class':"wikitable sortable"})
#pd.set_option('display.max_rows', None)

df=pd.read_html(str(indiatable))
# convert list to dataframe
df=pd.DataFrame(df[0])
print(df.head())
# Save dataframe to pickled pandas object
df.to_pickle("data.plk") # where to save it usually as a .plk
"""
# Load dataframe from pickled pandas object
df= pd.read_pickle("data.plk")

# drop the unwanted columns
data = df.drop(["Processor", "Designer", "MOS process(nm)", "Area (mm2)"], axis=1)

print("data shape",data.shape)

with pd.option_context('display.max_rows', None,):
    print(data)

######################################################################
#######     Data cleaning
#######     remove the citations and commas between numbers
y=df['MOS transistor count'].replace(to_replace=r"([0-99,\.]+)(.*)", value=r"\1", regex=True)
x=df['Date ofintroduction'].replace(to_replace=r"([0-99,\.]+)(.*)", value=r"\1", regex=True)
data=pd.concat([x, y], ignore_index=True, axis=1)
data = clean_data.replace(',','', regex=True)
data=data.dropna()
with pd.option_context('display.max_rows', None,):
    print(data)

data = data.drop(178)
clean_data = data.to_numpy()
clean_data = clean_data.astype(np.int64)
yrs=clean_data[:,0]
cnt=clean_data[:,1]

# taking log to convert the exponential data to linear
cnt=np.log(cnt)

X=yrs.reshape([yrs.size,1])
X = np.hstack((np.ones((X.size,1)), X))
Y=cnt
Y=Y.reshape(Y.size,1)
print("shapes", X.shape, Y.shape)
print(X)

theta = np.zeros((2, 1))
theta=np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))
#print(years)
#print(yrs)
#print(cnt)
#print("data = ",years)
#plt.scatter(yrs,cnt)
#plt.show()

# y = c + mx
fn=theta[0] + theta[1]*X[:,1]



plt.scatter(yrs, cnt)
plt.plot(X[:,1],fn,'r')
plt.xlabel('Years')
plt.ylabel('Count')
plt.legend(["Observations", "Linear regression model"], loc ="lower right")
plt.show()

