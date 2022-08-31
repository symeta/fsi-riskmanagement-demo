
```py
#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# In[4]:


raw = pd.read_csv("s3://fsi-assignment/creditcard.csv") #read data from S3
raw.head(10)


# In[5]:


#split the sample data into training set and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(raw.drop(['Class'],axis=1),raw['Class'],test_size = 0.3, random_state = 0)


# In[6]:


#define function to show 4 indicators of each model applied in order to evaluate them
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def show_score_comparison(_model,X_train,X_test,y_train,y_test,bar_width=0.2,opacity=0.8,color_train='b',color_test='g',fig_size_w=20,fig_size_h=8,mark_font_size=10):
    y_predict=_model.predict(X_test)
    y_predict_train=_model.predict(X_train)
    score_train=(accuracy_score(y_train,y_predict_train),
                 precision_score(y_train,y_predict_train),
                 recall_score(y_train,y_predict_train),
                 f1_score(y_train,y_predict_train))

    score_test=(accuracy_score(y_test,y_predict),
                precision_score(y_test,y_predict),
                recall_score(y_test,y_predict),
                f1_score(y_test,y_predict))

    plt.subplot(111)

    index=np.arange(len(score_train))

    rects1=plt.bar(index,score_train,bar_width,alpha=opacity,color=color_train,label='Train')
    mark_scores(score_train,mark_font_size=mark_font_size)

    rects2=plt.bar(index+bar_width,score_test,bar_width,alpha=opacity,color=color_test,label='Test')
    mark_scores(score_test,x_offset=bar_width,mark_font_size=mark_font_size)

    plt.xlabel('Score Type')
    plt.ylabel('Scores')
    plt.xticks(index+bar_width,('Accuracy','Precision','Recall','F1'))
    plt.yticks(list(np.arange(0.0,1.0,0.1)))
    plt.legend()
    plt.tight_layout()
    plt.gcf().set_size_inches(fig_size_w,fig_size_h)
    plt.show()

def mark_scores(scores,x_offset=0,mark_font_size=10):
    for each_score_index in range(len(scores)):
        plt.text(each_score_index+x_offset,scores[each_score_index]+0.05, '%.3f' % scores[each_score_index],
                 ha='center',va='bottom',fontsize=mark_font_size)


# In[7]:


#5 models are applied: Logistic Regresson, Random Forest, Gaussian NB, Decition Tree and Gradient Boosting 
lrc = LogisticRegression(C=0.01,penalty='l2')
lrc.fit(X_train,y_train.values.ravel())


# In[8]:


plt.figure(figsize=(10,5))
plt.title('Logistic Regression Model Score Comparison')
show_score_comparison(lrc,X_train,X_test,y_train,y_test)
plt.show()


# In[9]:


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


# In[10]:


plt.figure(figsize=(10,5))
plt.title('Random Forest Model Score Comparison')
show_score_comparison(rfc,X_train,X_test,y_train,y_test)
plt.show()


# In[11]:


gnb = GaussianNB()
gnb.fit(X_train, y_train)


# In[12]:


plt.figure(figsize=(10,5))
plt.title('Gaussian NB Model Score Comparison')
show_score_comparison(gnb,X_train,X_test,y_train,y_test)
plt.show()


# In[13]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)


# In[14]:


plt.figure(figsize=(10,5))
plt.title('Decision Tree Model Score Comparison')
show_score_comparison(dtc,X_train,X_test,y_train,y_test)
plt.show()


# In[15]:


gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)


# In[16]:


plt.figure(figsize=(10,5))
plt.title('Gradient Boosting Score Comparison')
show_score_comparison(gbc,X_train,X_test,y_train,y_test)
plt.show()


# In[17]:


p1 = lrc.predict_proba(X_test.values)
p2 = rfc.predict_proba(X_test.values)
p3 = gnb.predict_proba(X_test.values)
p4 = dtc.predict_proba(X_test.values)
p5 = gbc.predict_proba(X_test.values)


# In[18]:


from sklearn.metrics import roc_curve
fig = plt.figure(figsize=(10,8))
fpr1,tpr1,thre1 = roc_curve(y_test.values.ravel(),p1[:,1])
fpr2,tpr2,thre2 = roc_curve(y_test.values.ravel(),p2[:,1])
fpr3,tpr3,thre3 = roc_curve(y_test.values.ravel(),p3[:,1])
fpr4,tpr4,thre4 = roc_curve(y_test.values.ravel(),p4[:,1])
fpr5,tpr5,thre5 = roc_curve(y_test.values.ravel(),p5[:,1])

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr1,tpr1,label="LogisticRegression")
plt.plot(fpr2,tpr2,label="RandomForrest")
plt.plot(fpr3,tpr3,label="GaussianNB")
plt.plot(fpr4,tpr4,label="DecisionTree")
plt.plot(fpr5,tpr5,label="GradientBoosting")

plt.legend()


# In[ ]:


#in terms of model evaluation, two ways are applied: the indicator comparison, and the ROC/AUC
#from the result of the 4 indicator comparison, it is found out that Random Forrest Model has the best indicator value.
#from the ROC/AUC graph, GaussianNB has the best curve, while Random Forrest ranks the second
#however, the precision, recall, and F1 indicator of Gaussian NB is much lower than Random Forrest.
#as result, for this data sample, choose Random Forrest.


```
