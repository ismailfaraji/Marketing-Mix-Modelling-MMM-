#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_excel("C:/Users/hp/Desktop/MMM/MMM_DATA_Task.xlsx")
df.head()


# In[4]:


print(df.columns)
df.describe()


# In[6]:


corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))


# In[8]:


sns.pairplot(df)


# In[15]:


X = df.loc[:, df.columns != 'Sales']
y = df['Sales']


# In[32]:


import statsmodels.formula.api as sm
model = sm.ols(formula="Sales~Facebook_Impressions + Video_Impressions + Twitter_Impressions + Inflation + Display_Impressions + Average_Temp  + Offline_Investment + Competitor1_Investment + Competitor2_Investment + USD_RATE + EURO_RATE", data=df).fit()
print(model.summary())


# In[35]:


y_pred = model.predict()
labels = df['Sales']
df_temp = pd.DataFrame({'Actual': labels, 'Predicted':y_pred})
df_temp.head()


# In[37]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')


# In[39]:


y1 = df_temp['Actual']
y2 = df_temp['Predicted']


# In[40]:


plt.plot(y1, label = 'Actual')
plt.plot(y2, label = 'Predicted')
plt.legend()
plt.show()

