import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd

df = pd.read_csv('./kagglev2-may-2016.csv')

df = df.rename(columns = {'Hipertension': 'Hypertension', 'Handcap': 'Handicap', 'SMS_received': 'SMSReceived',
                          'No-show': 'NoShow'})

no_data = df[df.NoShow == 'No']
yes_data = df[df.NoShow == 'Yes']

no_data = no_data[:len(yes_data)]

df = pd.concat([no_data, yes_data])

df=df.sort_values(by=['AppointmentID'],axis=0)
df=df.reset_index(drop=True)
df['PatientId'] = df['PatientId'].astype('int64')
df = df[~(df['Age'] < 0)]    
df.Age.describe()
df = df.rename(columns={"NoShow": "OUTPUT_LABEL"})
df.OUTPUT_LABEL = df.OUTPUT_LABEL.map({ 'No': 0, 'Yes': 1 })

# Number of Appointments Missed by Patient
df['Num_App_Missed'] = df.groupby('PatientId')['OUTPUT_LABEL'].apply(lambda x: x.cumsum())

df['Num_App_Missed'] = df.Num_App_Missed- df.OUTPUT_LABEL        #모든 no-show 횟수에 -1
# df['Num_App_Missed']=np.where(df['Num_App_Missed']==0, 0, df['Num_App_Missed'] -1)


df.Num_App_Missed.replace(-1,0,inplace=True)                 #노쇼 횟수가 -1일 경우 0으로

# Number of Appointments Missed by Patient
df['TEMP'] = 1
df['Num_App'] = df.groupby('PatientId')['TEMP'].apply(lambda x: x.cumsum())
df['Num_App']=df.Num_App-1
df['Percent_Missed']=df['Num_App_Missed']/df['Num_App']
df.fillna(0, inplace=True)
df.drop(['TEMP'],axis=1, inplace=True)

df = df.replace('?',np.nan)

#Change to the date format
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'],format='%Y-%m-%d %H:%M:%S')
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'],format='%Y-%m-%d %H:%M:%S')
df['num_days'] = (df['AppointmentDay']-df['ScheduledDay']).dt.days
df.num_days = np.where(df.num_days<0, 0, df.num_days)
df.num_days.head(10)

df["day_of_week"] = df["ScheduledDay"].dt.dayofweek
df["month"] = df["ScheduledDay"].dt.month
df["week"] = df["ScheduledDay"].dt.week

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Neighbourhood'] = le.fit_transform(df['Neighbourhood'])

cols_num = ['Scholarship','Hypertension', 'Diabetes', 'Alcoholism','Neighbourhood',
       'SMSReceived', 'Age', 'num_days', 'day_of_week', 'month','week','Percent_Missed']

cols_cat = ['Gender']
cols_cat_num = ['Handicap']
df[cols_cat_num] = df[cols_cat_num].astype('str')
df_cat = pd.get_dummies(df[cols_cat + cols_cat_num],drop_first = True)   
df = pd.concat([df,df_cat], axis = 1)
cols_all_cat = list(df_cat.columns)
cols_input = cols_num + cols_all_cat
df_data = df[cols_input + ['OUTPUT_LABEL']]
dup_cols = set([x for x in cols_input if cols_input.count(x) > 1])
print(dup_cols)
assert len(dup_cols) == 0,'you have duplicated columns in cols_input'

assert (len(cols_input) + 1) == len(df_data.columns), 'issue with dimensions of df_data or cols_input'
# shuffle the samples
df_data = df_data.sample(n = len(df_data), random_state = 42)        #난수값 생성 변수(?)를 설정해주면 매번 같은 값이 나온다. (42인 이유 없음)
df_data = df_data.reset_index(drop = True)

# Save 30% of the data as validation and test data 
df_valid_test=df_data.sample(frac=0.30,random_state=42)
print('Split size: %.3f'%(len(df_valid_test)/len(df_data)))
#And now split into test and validation using 50% fraction
df_test = df_valid_test.sample(frac = 0.5, random_state = 42)
df_valid = df_valid_test.drop(df_test.index)

# use the rest of the data as training data
df_train_all=df_data.drop(df_valid_test.index)
# Number of Appointments Missed by Patient

def calc_prevalence(y_actual):
    # this function calculates the prevalence of the positive class (label = 1)
    return (sum(y_actual)/len(y_actual))

# split the training data into positive and negative
rows_pos = df_train_all.OUTPUT_LABEL == 1
df_train_pos = df_train_all.loc[rows_pos]
df_train_neg = df_train_all.loc[~rows_pos]

n = np.min([len(df_train_pos), len(df_train_neg)])

# merge the balanced data
df_train = pd.concat([df_train_pos.sample(n = n, random_state = 42), 
                      df_train_neg.sample(n = n, random_state = 42)],axis = 0, 
                     ignore_index = True)

# shuffle the order of training samples 
df_train = df_train.sample(n = len(df_train), random_state = 42).reset_index(drop = True)

print('Train balanced prevalence(n = %d):%.3f'%(len(df_train), calc_prevalence(df_train.OUTPUT_LABEL.values)))

# create the X and y matrices
X_train = df_train[cols_input].values
X_train_all = df_train_all[cols_input].values
X_valid = df_valid[cols_input].values

y_train = df_train['OUTPUT_LABEL'].values
y_valid = df_valid['OUTPUT_LABEL'].values

from sklearn.preprocessing import StandardScaler
scaler  = StandardScaler()
scaler.fit(X_train_all)

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
def calc_specificity(y_actual, y_pred, thresh):
    # calculates specificity
    return sum((y_pred < thresh) & (y_actual == 0)) /sum(y_actual ==0)

def print_report(y_actual, y_pred, thresh):
    

    accuracy = accuracy_score(y_actual, (y_pred > thresh))

    print('accuracy:%.3f'%accuracy)

    print(' ')
    return accuracy
thresh = 0.5



def fill_my_missing(df, df_mean, col2use):
    # This function fills the missing values

    # check the columns are present
    for c in col2use:
        assert c in df.columns, c + ' not in df'
        assert c in df_mean.col.values, c+ 'not in df_mean'
    
    # replace the mean 
    for c in col2use:
        mean_value = df_mean.loc[df_mean.col == c,'mean_val'].values[0]
        df[c] = df[c].fillna(mean_value)
    return df
# your code here
df_mean = df_train_all[cols_input].mean(axis = 0)
# save the means
df_mean.to_csv('df_mean.csv',index=True)
df_mean_in = pd.read_csv('df_mean.csv', names =['col','mean_val'])
df_mean_in.head()

# fill missing
df_train = fill_my_missing(df_train, df_mean_in, cols_input)
df_valid = fill_my_missing(df_valid, df_mean_in, cols_input)
df_test = fill_my_missing(df_test, df_mean_in, cols_input)

# create X and y matrices
X_train = df_train[cols_input].values
X_valid = df_valid[cols_input].values
X_test = df_test[cols_input].values

y_train = df_train['OUTPUT_LABEL'].values
y_valid = df_valid['OUTPUT_LABEL'].values
y_test = df_test['OUTPUT_LABEL'].values

# transform our data matrices 
X_train_tf = scaler.transform(X_train)
X_valid_tf = scaler.transform(X_valid)
X_test_tf = scaler.transform(X_test)

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

lr_train_accuracy=[0]*50
lr_valid_accuracy=[0]*50
lr_test_accuracy=[0]*50
tree_train_accuracy=[0]*50
tree_valid_accuracy=[0]*50
tree_test_accuracy=[0]*50
rf_train_accuracy=[0]*50
rf_valid_accuracy=[0]*50
rf_test_accuracy=[0]*50
for idx in range(50):
    lr=LogisticRegression(random_state = idx)
    lr.fit(X_train_tf, y_train)

    y_train_preds = lr.predict_proba(X_train_tf)[:,1]
    y_valid_preds = lr.predict_proba(X_valid_tf)[:,1]
    y_test_preds = lr.predict_proba(X_test_tf)[:,1]

    print('Logistic Regression')
    print('Training:')
    lr_train_accuracy[idx] = print_report(y_train,y_train_preds, thresh)
    print('Validation:')
    lr_valid_accuracy[idx] = print_report(y_valid,y_valid_preds, thresh)
    print('Test:')
    lr_test_accuracy[idx] = print_report(y_test,y_test_preds, thresh)

    tree = DecisionTreeClassifier(max_depth = 10, random_state = idx)
    tree.fit(X_train_tf, y_train)

    y_train_preds = tree.predict_proba(X_train_tf)[:,1]
    y_valid_preds = tree.predict_proba(X_valid_tf)[:,1]
    y_test_preds = tree.predict_proba(X_test_tf)[:,1]

    print('Decision Tree')
    print('Training:')
    tree_train_accuracy[idx] =print_report(y_train,y_train_preds, thresh)
    print('Validation:')
    tree_valid_accuracy[idx] = print_report(y_valid,y_valid_preds, thresh)
    print('Testing:')
    tree_test_accuracy[idx] = print_report(y_test,y_test_preds, thresh)

    rf=RandomForestClassifier(max_depth = 6, random_state = idx)
    rf.fit(X_train_tf, y_train)

    y_train_preds = rf.predict_proba(X_train_tf)[:,1]
    y_valid_preds = rf.predict_proba(X_valid_tf)[:,1]
    y_test_preds = rf.predict_proba(X_test_tf)[:,1]

    print('Random Forest')
    print('Training:')
    rf_train_accuracy[idx] =print_report(y_train,y_train_preds, thresh)
    print('Validation:')
    rf_valid_accuracy[idx] = print_report(y_valid,y_valid_preds, thresh)
    print('Testing:')
    rf_test_accuracy[idx] = print_report(y_test,y_test_preds, thresh)

print('train: ', np.mean(lr_train_accuracy), np.mean(tree_train_accuracy), np.mean(rf_train_accuracy))
print('valid: ', np.mean(lr_valid_accuracy), np.mean(tree_valid_accuracy), np.mean(rf_valid_accuracy))
print('test: ', np.mean(lr_test_accuracy), np.mean(tree_test_accuracy), np.mean(rf_test_accuracy))