# fsi-riskmanagement-demo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
raw = pd.read_csv("s3://fsi-assignment/creditcard.csv") #read data from S3
raw.head(10)
Time	V1	V2	V3	V4	V5	V6	V7	V8	V9	...	V21	V22	V23	V24	V25	V26	V27	V28	Amount	Class
0	0.0	-1.359807	-0.072781	2.536347	1.378155	-0.338321	0.462388	0.239599	0.098698	0.363787	...	-0.018307	0.277838	-0.110474	0.066928	0.128539	-0.189115	0.133558	-0.021053	149.62	0
1	0.0	1.191857	0.266151	0.166480	0.448154	0.060018	-0.082361	-0.078803	0.085102	-0.255425	...	-0.225775	-0.638672	0.101288	-0.339846	0.167170	0.125895	-0.008983	0.014724	2.69	0
2	1.0	-1.358354	-1.340163	1.773209	0.379780	-0.503198	1.800499	0.791461	0.247676	-1.514654	...	0.247998	0.771679	0.909412	-0.689281	-0.327642	-0.139097	-0.055353	-0.059752	378.66	0
3	1.0	-0.966272	-0.185226	1.792993	-0.863291	-0.010309	1.247203	0.237609	0.377436	-1.387024	...	-0.108300	0.005274	-0.190321	-1.175575	0.647376	-0.221929	0.062723	0.061458	123.50	0
4	2.0	-1.158233	0.877737	1.548718	0.403034	-0.407193	0.095921	0.592941	-0.270533	0.817739	...	-0.009431	0.798278	-0.137458	0.141267	-0.206010	0.502292	0.219422	0.215153	69.99	0
5	2.0	-0.425966	0.960523	1.141109	-0.168252	0.420987	-0.029728	0.476201	0.260314	-0.568671	...	-0.208254	-0.559825	-0.026398	-0.371427	-0.232794	0.105915	0.253844	0.081080	3.67	0
6	4.0	1.229658	0.141004	0.045371	1.202613	0.191881	0.272708	-0.005159	0.081213	0.464960	...	-0.167716	-0.270710	-0.154104	-0.780055	0.750137	-0.257237	0.034507	0.005168	4.99	0
7	7.0	-0.644269	1.417964	1.074380	-0.492199	0.948934	0.428118	1.120631	-3.807864	0.615375	...	1.943465	-1.015455	0.057504	-0.649709	-0.415267	-0.051634	-1.206921	-1.085339	40.80	0
8	7.0	-0.894286	0.286157	-0.113192	-0.271526	2.669599	3.721818	0.370145	0.851084	-0.392048	...	-0.073425	-0.268092	-0.204233	1.011592	0.373205	-0.384157	0.011747	0.142404	93.20	0
9	9.0	-0.338262	1.119593	1.044367	-0.222187	0.499361	-0.246761	0.651583	0.069539	-0.736727	...	-0.246914	-0.633753	-0.120794	-0.385050	-0.069733	0.094199	0.246219	0.083076	3.68	0
10 rows × 31 columns

#split the sample data into training set and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(raw.drop(['Class'],axis=1),raw['Class'],test_size = 0.3, random_state = 0)
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
#5 models are applied: Logistic Regresson, Random Forest, Gaussian NB, Decition Tree and Gradient Boosting 
lrc = LogisticRegression(C=0.01,penalty='l2')
lrc.fit(X_train,y_train.values.ravel())
/home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:765: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
LogisticRegression(C=0.01)
plt.figure(figsize=(10,5))
plt.title('Logistic Regression Model Score Comparison')
show_score_comparison(lrc,X_train,X_test,y_train,y_test)
plt.show()
/home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/ipykernel/__main__.py:17: MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
RandomForestClassifier()
plt.figure(figsize=(10,5))
plt.title('Random Forest Model Score Comparison')
show_score_comparison(rfc,X_train,X_test,y_train,y_test)
plt.show()
/home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/ipykernel/__main__.py:17: MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.

gnb = GaussianNB()
gnb.fit(X_train, y_train)
GaussianNB()
plt.figure(figsize=(10,5))
plt.title('Gaussian NB Model Score Comparison')
show_score_comparison(gnb,X_train,X_test,y_train,y_test)
plt.show()
/home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/ipykernel/__main__.py:17: MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
DecisionTreeClassifier()
plt.figure(figsize=(10,5))
plt.title('Decision Tree Model Score Comparison')
show_score_comparison(dtc,X_train,X_test,y_train,y_test)
plt.show()
/home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/ipykernel/__main__.py:17: MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.

gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
GradientBoostingClassifier()
plt.figure(figsize=(10,5))
plt.title('Gradient Boosting Score Comparison')
show_score_comparison(gbc,X_train,X_test,y_train,y_test)
plt.show()
/home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/ipykernel/__main__.py:17: MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.

p1 = lrc.predict_proba(X_test.values)
p2 = rfc.predict_proba(X_test.values)
p3 = gnb.predict_proba(X_test.values)
p4 = dtc.predict_proba(X_test.values)
p5 = gbc.predict_proba(X_test.values)
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
<matplotlib.legend.Legend at 0x7f7908d7ecc0>

#in terms of model evaluation, two ways are applied: the indicator comparison, and the ROC/AUC
#from the result of the 4 indicator comparison, it is found out that Random Forrest Model has the best indicator value.
#from the ROC/AUC graph, GaussianNB has the best curve, while Random Forrest ranks the second
#however, the precision, recall, and F1 indicator of Gaussian NB is much lower than Random Forrest.
#as result, for this data sample, choose Random Forrest.
