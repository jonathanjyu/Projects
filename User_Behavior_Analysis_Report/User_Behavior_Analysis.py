#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


def read_csv():
    # read csv
    df = pd.read_csv('uselog.csv')
    print(df.info())
    # turn timestamps to type datetime
    df['timestamps of usage'] = pd.to_datetime(df['timestamps of usage'])
    print(df.info())
    
    #drop na
    df = df.dropna()
    print(df.info())
    return df


# In[3]:


df = read_csv()


# In[4]:


print(df.head(10))


# ## observe ID

# ### How many IDs / user types / place of residence / function

# In[5]:


len(df['ID'].unique())


# In[6]:


len(df['user type'].unique())


# In[7]:


len(df['place of residence'].unique())


# In[8]:


len(df['function'].unique())


# ### about timestamps

# In[9]:


sorted(df['timestamps of usage'])[0],sorted(df['timestamps of usage'])[-1]


# ### observe each column counts

# In[10]:


# 計算每個user type的數量
UT_counts = df['user type'].value_counts()
print(UT_counts)


# In[11]:


# 計算每個place of residence的數量
PoR_counts = df['place of residence'].value_counts()
print(PoR_counts)


# In[12]:


# 計算每個function的數量
F_counts = df['function'].value_counts()
print(F_counts)


# ## observe groups distribution

# ### observer user type

# In[13]:


# 設定顏色
colors = ['lightblue', 'lightcoral', 'lightgreen']


# In[14]:


# 繪製bar
plt.bar(UT_counts.index, UT_counts.values, tick_label=UT_counts.index,color = colors)

# 在每個bar上顯示數量
for i, height in enumerate(UT_counts.values):
    plt.text(UT_counts.index[i], height, str(height), ha='center')

# 設定圖表標題和軸標籤
plt.title('User Type Distribution')
plt.xlabel('User Type')
plt.ylabel('Count')
plt.savefig('User_Type_Distribution.png', bbox_inches = 'tight')

# 顯示圖表
plt.show()


# In[15]:


# 繪製圓餅圖
plt.pie(UT_counts.values, labels=UT_counts.index, autopct='%1.2f%%',colors = colors)
plt.title('User Type Percentage')
plt.savefig('User_Type_Percentage.png', bbox_inches = 'tight')


# ### observer place of residence

# In[16]:


# 設定顏色
colors = ['red', 'orange', 'yellow', 'green']


# In[17]:


# 繪製bar
plt.bar(PoR_counts.index, PoR_counts.values, tick_label=PoR_counts.index,color = colors)

# 在每個bar上顯示數量
for i, height in enumerate(PoR_counts.values):
    plt.text(PoR_counts.index[i], height, str(height), ha='center')

# 設定圖表標題和軸標籤
plt.title('place of residence Distribution')
plt.xlabel('place of residence')
plt.ylabel('Count')
plt.savefig('place_of_residence_Distribution.png', bbox_inches = 'tight')

# 顯示圖表
plt.show()


# In[18]:


# 繪製圓餅圖
plt.pie(PoR_counts.values, labels=PoR_counts.index, autopct='%1.2f%%',colors = colors)
plt.title('place of residence Percentage')
plt.savefig('place_of_residence_Percentage.png', bbox_inches = 'tight')


# ### observer function

# In[19]:


# 設定顏色
# 隨機產生 13 種顏色
import random
colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(13)]


# In[20]:


# 繪製bar
plt.figure(figsize=(20, 6))
plt.bar(F_counts.index, F_counts.values, tick_label=F_counts.index,color = colors)

# 在每個bar上顯示數量
for i, height in enumerate(F_counts.values):
    if height == max(F_counts.values):
        plt.text(F_counts.index[i], height, str(height), ha='center',fontsize = 20, color = 'red')
        
    else:
        plt.text(F_counts.index[i], height, str(height), ha='center',fontsize = 10)

# 設定圖表標題和軸標籤
plt.title('function Distribution',fontsize = 20)
plt.xlabel('function',fontsize = 20)
plt.ylabel('Count',fontsize = 20)
plt.xticks(rotation=90,fontsize = 20)


# 顯示圖表
plt.savefig('function_Distribution_bar.png', bbox_inches = 'tight')
plt.show()


# ## user type analysis

# ### observe timestamps of usage

# In[21]:


df = read_csv()
df.set_index('timestamps of usage', inplace=True)

plt.figure(figsize = (10,6))
plt.plot(df.resample('D').count(),color='royalblue')
plt.title('User Activity')
plt.xlabel('Date')
plt.ylabel('Usage Count')
plt.savefig('User_Activity.png', bbox_inches = 'tight')
plt.show()


# ### observe daily usage

# In[22]:


## 每小時所用功能使用次數


# In[23]:


df = read_csv()

# 將timestamps of usage轉換為datetime類型
df['timestamps of usage'] = pd.to_datetime(df['timestamps of usage'])

# 按照時間分組，計算每個小時的功能使用次數的總和
usage_by_time = df.groupby(df['timestamps of usage'].dt.hour)['function'].count()

# plot折線圖
plt.figure(figsize = (10,6))
plt.plot(usage_by_time, markersize="16", marker=".")
plt.title('Usage by hour')
plt.xlabel('Time (each hour)')
plt.ylabel('Usage Count')
plt.grid(axis = 'x')
plt.xticks(ticks = range(0,24))
plt.savefig('Usage_by_hour.png', bbox_inches = 'tight')
plt.show()


# In[24]:


## 依user type分類每小時所有功能使用次數


# In[25]:


# 使用pivot_table()方法將資料進行轉換

usage_by_time_and_user_type = df.pivot_table(index=df['timestamps of usage'].dt.hour, columns='user type', values='function', aggfunc='count')

# 繪製折線圖
plt.figure(figsize = (10,6))
plt.plot(usage_by_time_and_user_type['A'],color = 'lightblue', markersize="16", marker=".")
plt.plot(usage_by_time_and_user_type['L'],color = 'lightgreen', markersize="16", marker=".")
plt.plot(usage_by_time_and_user_type['S'],color = 'lightcoral', markersize="16", marker=".")
plt.title('Usage by hour groupby user type')
plt.xlabel('Time (each hour)')
plt.ylabel('Usage Count')
plt.grid(axis = 'x')
plt.legend(usage_by_time_and_user_type.columns)
plt.xticks(ticks = range(0,24))
plt.savefig('Usage_by_hour_groupby_usertype.png', bbox_inches = 'tight')
plt.show()


# In[26]:


## 依place of residence分類每小時所有功能使用次數


# In[27]:


# 使用pivot_table()方法將資料進行轉換
usage_by_time_and_residence = df.pivot_table(index=df['timestamps of usage'].dt.hour, columns='place of residence', values='function', aggfunc='count')

# 繪製折線圖
plt.figure(figsize = (10,6))
plt.plot(usage_by_time_and_residence['P01'],color = 'yellow', markersize="16", marker=".")
plt.plot(usage_by_time_and_residence['P02'],color = 'orange', markersize="16", marker=".")
plt.plot(usage_by_time_and_residence['P03'],color = 'red', markersize="16", marker=".")
plt.plot(usage_by_time_and_residence['P05'],color = 'green', markersize="16", marker=".")
plt.title('Usage by hour groupby place of residence')
plt.xlabel('Time (each hour)')
plt.ylabel('Usage Count')
plt.grid(axis = 'x')
plt.legend(usage_by_time_and_residence.columns)
plt.xticks(ticks = range(0,24))
plt.savefig('Usage_by_hour_groupby_placeofresidence.png', bbox_inches = 'tight')
plt.show()


# In[28]:


## 每小時被使用最多次的功能及其使用次數


# In[29]:


df['hour'] = df['timestamps of usage'].dt.hour

# 依每小時統計各功能使用次數
hourly_function_count = df.groupby(['hour', 'function']).size().reset_index(name='count')

# 找出每小時使用最多的功能及其使用次數
max_function_per_hour = hourly_function_count.loc[hourly_function_count.groupby('hour')['count'].idxmax()]

# 繪製長條圖
plt.figure(figsize=(24, 12))
plt.bar(max_function_per_hour['hour'], max_function_per_hour['count'], color='skyblue')
plt.xlabel('Time (each hour)',fontsize = 20)
plt.ylabel('Usage Count',fontsize = 20)
plt.title('Most Used Function(count) by Hour',fontsize = 20)
plt.xticks(range(24),fontsize = 20)

# 在每個bar上顯示該功能及數量
for index, row in max_function_per_hour.iterrows():
    if row['count'] == max(max_function_per_hour['count']):
        plt.text(row['hour'], row['count'], f"{row['function']} ({row['count']})", ha='center', va='bottom',fontsize = 20,color = 'red')
    else:
        plt.text(row['hour'], row['count'], f"{row['function']} ({row['count']})", ha='center', va='bottom',fontsize = 14)

plt.savefig('Most_Used_Function(count)_by_Hour.png', bbox_inches = 'tight')
    
plt.grid(axis='y')
plt.show()


# ### observer each function usage per hour

# In[30]:


# function 'F13'
function_F13 = hourly_function_count[hourly_function_count['function'] == 'F13']

# function 'F01' 在每小時的使用次數
plt.figure(figsize=(10, 6))
plt.bar(function_F13['hour'], function_F13['count'], color='bisque')
plt.xlabel('Time (each hour)')
plt.ylabel('Usage Count')
plt.title('Usage of Function F13 each Hour')
plt.xticks(range(24))

for index, row in function_F13.iterrows():
    plt.text(row['hour'], row['count'], row['count'], ha='center', va='bottom')

plt.savefig('Usage_of_Function_F13_each_Hour.png', bbox_inches = 'tight')
plt.grid(axis='y')
plt.show()


# ### observe weekly usage

# In[31]:


min(df['timestamps of usage']),max(df['timestamps of usage'])


# In[32]:


# turn timestamps to day of week
df['day_of_week'] = df['timestamps of usage'].dt.dayofweek
weekday_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
df['day_of_week'] = df['day_of_week'].map(weekday_map)
df


# In[33]:


# 按照星期分組，計算每周的功能使用次數的總和
usage_by_weekly = df.groupby(df['day_of_week'])['function'].count()

weekdays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# 使用 reindex 方法重新排序 usage_by_weekly
usage_by_weekly_sorted = usage_by_weekly.reindex(weekdays_order)
print(usage_by_weekly_sorted)

# plot折線圖
plt.figure(figsize = (10,6))
plt.plot(usage_by_weekly_sorted, marker=".", markersize="16")
plt.title('Usage by day')
plt.xlabel('Day (for week)')
plt.ylabel('Usage Count')
plt.grid(axis = 'x')
plt.xticks(ticks = range(0,7))

plt.savefig('Usage_by_day.png', bbox_inches = 'tight')
plt.show()


# ### observe monthly usage

# In[34]:


# turn timestamps to month
df['month'] = df['timestamps of usage'].dt.month
# 將月份映射到實際的月份名稱
month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

df['month'] = df['month'].map(month_map)
df


# In[35]:


# 按照星期分組，計算每周的功能使用次數的總和
usage_by_month = df.groupby(df['month'])['function'].count()

month_order = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# 使用 reindex 方法重新排序 usage_by_weekly
usage_by_month_sorted = usage_by_month.reindex(month_order)
print(usage_by_month_sorted)

# plot折線圖
plt.figure(figsize = (10,6))
plt.plot(usage_by_month_sorted, marker=".", markersize="16")
plt.title('Usage by month')
plt.xlabel('Month')
plt.ylabel('Usage Count')
plt.grid(axis = 'x')
plt.xticks(ticks = range(4,7))

plt.savefig('Usage_by_month.png', bbox_inches = 'tight')
plt.show()


# # Machine Learning Analysis

# In[36]:


import numpy as np
from sklearn.model_selection import train_test_split

#import model
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#evaluation
from sklearn import metrics


# In[37]:


df = read_csv()


# ## Classification

# ### 1.predict user type

# ### data preprocessing

# In[38]:


# encode timestamps to date, day of week, hour
df['timestamps_hour'] = df['timestamps of usage'].dt.hour
df['timestamps_day'] = df['timestamps of usage'].dt.dayofweek
df['timestamps_date'] = df['timestamps of usage'].dt.day

df = df.drop('timestamps of usage', axis = 1)
df = df.drop('ID', axis = 1)


# In[39]:


# label encoding user type
usertype_mapping = {'A':0, 'S':1, 'L':2} #define map as dictionary
df['user type'] = df['user type'].map(usertype_mapping)

#encoding categorical features
df = pd.get_dummies(df).reset_index(drop=True)


# In[40]:


df


# In[41]:


# We predict user type first
y = df[['user type']]
df.drop(['user type'], axis=1, inplace=True)
x = df


# In[42]:


# train test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)


# In[43]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[44]:


## create list to save the accuracy of each model prediction
predAccList_userType = []


# In[45]:


# create KNN Classifier model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
model.fit(X_train, y_train)

# use the model to predict the labels of the test data
y_predict = model.predict(X_test)

# accuracy
acc = metrics.accuracy_score(y_true=y_test, y_pred=y_predict)
predAccList_userType.append(acc)

print('accuracy: ', acc)
print(metrics.classification_report(y_test, y_predict))


# In[46]:


# create XGB Classifier model
model = XGBClassifier()

# Train the model using the training sets
model.fit(X_train, y_train)

# use the model to predict the labels of the test data
y_predict = model.predict(X_test)

# accuracy
acc = metrics.accuracy_score(y_true=y_test, y_pred=y_predict)
predAccList_userType.append(acc)

print('accuracy: ', acc)
print(metrics.classification_report(y_test, y_predict))


# In[47]:


# create DecisionTreeClassifier model
model = DecisionTreeClassifier()

# Train the model using the training sets
model.fit(X_train, y_train)

# use the model to predict the labels of the test data
y_predict = model.predict(X_test)

# accuracy
acc = metrics.accuracy_score(y_true=y_test, y_pred=y_predict)
predAccList_userType.append(acc)

print('accuracy: ', acc)
print(metrics.classification_report(y_test, y_predict))


# In[48]:


# create RandomForestClassifier model
model = RandomForestClassifier()

# Train the model using the training sets
model.fit(X_train, y_train)

# use the model to predict the labels of the test data
y_predict = model.predict(X_test)

# accuracy
acc = metrics.accuracy_score(y_true=y_test, y_pred=y_predict)
predAccList_userType.append(acc)

print('accuracy: ', acc)
print(metrics.classification_report(y_test, y_predict))


# In[49]:


acc = {
    'models':['KNN', 'XGB', 'DecisionTree', 'RandomForest'],
    'accuracy':predAccList_userType
}
acc = pd.DataFrame(acc)

plt.bar(acc['models'], acc['accuracy'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('[\'user type\'] Prediction Accuracy Comparison of Different Models')
for index, row in acc.iterrows():
    plt.text(row['models'], row['accuracy'], '{:.2%}'.format(row['accuracy']), ha='center', va='bottom')

plt.savefig('user_type_MLpred.png', bbox_inches = 'tight')
plt.show()


# ### 2.predict place of residence

# In[50]:


df = read_csv()

# encode timestamps to date, day of week, hour
df['timestamps_hour'] = df['timestamps of usage'].dt.hour
df['timestamps_day'] = df['timestamps of usage'].dt.dayofweek
df['timestamps_date'] = df['timestamps of usage'].dt.day

df = df.drop('timestamps of usage', axis = 1)
df = df.drop('ID', axis = 1)

# label encoding function
placeofres_mapping = {'P01':1,'P02':2,'P03':3,'P04':4,'P05':5} #define map as dictionary
df['place of residence'] = df['place of residence'].map(placeofres_mapping)

#encoding categorical features
df = pd.get_dummies(df).reset_index(drop=True)


# In[51]:


# We predict place of residence
y = df[['place of residence']]
df.drop(['place of residence'], axis=1, inplace=True)
x = df

# train test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[52]:


## create list to save the accuracy of each model prediction
predAccList_placeofresidence = []


# In[53]:


# create KNN Classifier model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
model.fit(X_train, y_train)

# use the model to predict the labels of the test data
y_predict = model.predict(X_test)

# accuracy
acc = metrics.accuracy_score(y_true=y_test, y_pred=y_predict)
predAccList_placeofresidence.append(acc)

print('accuracy: ', acc)
print(metrics.classification_report(y_test, y_predict))


# In[54]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train2 = le.fit_transform(y_train)
y_train2
y_test2 = le.fit_transform(y_test)
y_test2


# In[55]:


# create XGB Classifier model
model = XGBClassifier()

# Train the model using the training sets
model.fit(X_train, y_train2)

# use the model to predict the labels of the test data
y_predict = model.predict(X_test)

# accuracy
acc = metrics.accuracy_score(y_true=y_test2, y_pred=y_predict)
predAccList_placeofresidence.append(acc)

print('accuracy: ', acc)
print(metrics.classification_report(y_test2, y_predict))


# In[56]:


# create DecisionTreeClassifier model
model = DecisionTreeClassifier()

# Train the model using the training sets
model.fit(X_train, y_train)

# use the model to predict the labels of the test data
y_predict = model.predict(X_test)

# accuracy
acc = metrics.accuracy_score(y_true=y_test, y_pred=y_predict)
predAccList_placeofresidence.append(acc)

print('accuracy: ', acc)
print(metrics.classification_report(y_test, y_predict))


# In[57]:


# create RandomForestClassifier model
model = RandomForestClassifier()

# Train the model using the training sets
model.fit(X_train, y_train)

# use the model to predict the labels of the test data
y_predict = model.predict(X_test)

# accuracy
acc = metrics.accuracy_score(y_true=y_test, y_pred=y_predict)
predAccList_placeofresidence.append(acc)

print('accuracy: ', acc)
print(metrics.classification_report(y_test, y_predict))


# In[58]:


acc = {
    'models':['KNN', 'XGB', 'DecisionTree', 'RandomForest'],
    'accuracy':predAccList_placeofresidence
}
acc = pd.DataFrame(acc)

plt.bar(acc['models'], acc['accuracy'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('[\'place of residence\'] Prediction Accuracy Comparison of Different Models')
for index, row in acc.iterrows():
    plt.text(row['models'], row['accuracy'], '{:.2%}'.format(row['accuracy']), ha='center', va='bottom')
    

plt.savefig('place_of_residence_MLpred.png', bbox_inches = 'tight')
plt.show()


# In[ ]:




