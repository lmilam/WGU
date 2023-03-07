#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Standard data science imports
import numpy as np
import pandas as pd
from pandas import DataFrame

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Statistics packages
import pylab
import statsmodels.api as sm
import statistics
from scipy import stats

# Import chisquare from SciPy.stats
from scipy.stats import chisquare
from scipy.stats import chi2_contingency


# In[2]:


# Load data set into Pandas dataframe
df = pd.read_csv('medical_clean.csv')
df.columns


# In[3]:


# Rename survey columns to more identifiable names
df.rename(columns = 
    {'Item1':'Survey_TimelyAdmin',
     'Item2':'Survey_TimelyTreatment',
     'Item3':'Survey_TimelyVisits',
     'Item4':'Survey_Reliability',
     'Item5':'Survey_Options',
     'Item6':'Survey_HoursTreatment',
     'Item7':'Survey_CourteousStaff',
     'Item8':'Survey_ActiveListening'}, inplace=True)


# In[4]:


contingency = pd.crosstab(df['ReAdmis'], df['Survey_TimelyAdmin'])
contingency


# In[5]:


contingency_pct = pd.crosstab(df['ReAdmis'], df['Survey_TimelyAdmin'], normalize='index')
contingency_pct


# In[6]:


plt.figure(figsize=(12,8))
sns.heatmap(contingency, annot=True, cmap="YlGnBu")


# In[7]:


# Chi-square test of independence
c, p, dof, expected = chi2_contingency(contingency)
print('p-value = ' + str(p))


# In[8]:


df.describe()


# In[9]:


# Histograms of continous and categorical variables
df[['TotalCharge','Additional_charges','Survey_TimelyAdmin','Survey_CourteousStaff']].hist()
plt.tight_layout()


# In[10]:


# Seaborn boxplots of continous and categorical variables 
sns.boxplot('TotalCharge', data = df)
plt.show()


# In[11]:


sns.boxplot('Additional_charges', data = df)
plt.show()


# In[12]:


sns.boxplot('Survey_TimelyAdmin', data = df)
plt.show()


# In[13]:


sns.boxplot('Survey_CourteousStaff', data = df)
plt.show()


# In[14]:


# Dataframe for heatmap for bivariate analysis of correlation
readmis_bivariate = df[['TotalCharge','Additional_charges','Survey_TimelyAdmin','Survey_CourteousStaff']]


# In[15]:


sns.heatmap(readmis_bivariate.corr(),annot=True)
plt.show()


# In[16]:


# Scatter plot of continuous variables TotalCharge and Additional_charges
readmis_bivariate[readmis_bivariate['TotalCharge'] < 10000].sample(100).plot.scatter(x = 'TotalCharge',y = 'Additional_charges')

# Scatter plot of categorical variables Survey_TimelyAdmin and Survey_CourteousStaff
readmis_bivariate[readmis_bivariate['Survey_TimelyAdmin'] < 10].sample(100).plot.scatter(x = 'Survey_TimelyAdmin',y = 'Survey_CourteousStaff')


# In[17]:


# Heatmap for continous variables TotalCharge and Additional_charges
readmis_bivariate[readmis_bivariate['TotalCharge'] < 10000].plot.hexbin(x = 'TotalCharge', y = 'Additional_charges')


# In[ ]:




