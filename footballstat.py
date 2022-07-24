import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from google.colab import drive 

data_set = pd.read_csv('teamstats.csv')
x=pd.read_csv('teamstats.csv', usecols=["shots"])
y=pd.read_csv('teamstats.csv', usecols=['fouls'])

model = LinearRegression()
model.fit(x,y)
plt.figure()
plt.title('Shots vs fouls statistics')
plt.xlabel('Shots (in number)')
plt.ylabel('Fouls (in number)')
plt.plot(x,y,'.')
plt.plot(x,model.predict(x),'--')
plt.axis([0,25,0,25])
plt.grid(True)
print ("Predicted Value = ",model.predict([[21]])) # 22.467
plt.show()
