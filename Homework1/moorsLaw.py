import pandas as pd # library for data analysis
import re
import requests # library to handle requests
from bs4 import BeautifulSoup # library to parse HTML documents
import numpy as np
import matplotlib.pyplot as plt

p = np.zeros((2, 1))
""""
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
# rename columns for ease
#data = data.rename(columns={"State or union territory": "State","Population(2011)[3]": "Population"})
print("data shape",data.shape)
#pd.set_option('display.max_rows', 10)

#d.set_option('display.max_rows', data.shape[0]+1)
#print(data.head())

with pd.option_context('display.max_rows', None,):
    print(data)

#extr = data['Date ofintroduction'].str.extract(r'^(\d{4})', expand=False)
#print(extr.head())
#with pd.option_context('display.max_rows', None,):
#    print(extr)

#regex = re.compile(".*?\((.*?)\)")
#result = re.findall(regex, data)
#dat=re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", data)
#print(result)
y=df['MOS transistor count'].replace(to_replace=r"([0-99,\.]+)(.*)", value=r"\1", regex=True)
x=df['Date ofintroduction'].replace(to_replace=r"([0-99,\.]+)(.*)", value=r"\1", regex=True)
z=pd.concat([x, y], ignore_index=True, axis=1)
F = z.replace(',','', regex=True)
G=F.dropna()
with pd.option_context('display.max_rows', None,):
    print(G)

G = G.drop(178)
years = G.to_numpy()
years = years.astype(np.int64)
xxxx=years[:,0]
yyyy=years[:,1]


X=xxxx.reshape([xxxx.size,1])
X = np.hstack((np.ones((X.size,1)), X))
Y=yyyy
Y=Y.reshape(Y.size,1)
print("shapes>>>>>>>>>>>>>>", X.shape, Y.shape)
print(X)

p=np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))
#print(years)
#print(xxxx)
#print(yyyy)
#print("data = ",years)
#plt.scatter(xxxx,yyyy)
#plt.show()

# y = c + mx
fn=p[0] + p[1]*X[:,1]



plt.scatter(xxxx, yyyy)
plt.plot(X[:,1],fn,'r')
plt.xlabel('Years')
plt.ylabel('Count')
plt.legend(["Observations", "Linear regression model"], loc ="lower right")
plt.show()

