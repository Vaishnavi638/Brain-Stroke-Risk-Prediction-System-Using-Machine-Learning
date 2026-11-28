#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


train_df = pd.read_csv("Stroke_train.csv")
test_df = pd.read_csv("Stroke_test.csv")
sub_df = pd.read_csv("Stroke_sample_submission.csv")


# In[3]:


train_df.head()


# In[4]:


train_df['stroke'].unique()


# In[5]:


train_df.info()


# In[6]:


train_df.describe()


# In[7]:


train_df.nunique()


# ### Data Cleaning

# In[9]:


train_df.duplicated().sum()


# In[10]:


sorted(train_df['age'].unique().tolist())[0:12]


# In[11]:


train_df['age'] = np.ceil(train_df['age'])
test_df['age'] = np.ceil(test_df['age'])


# In[12]:


train_df['gender'].value_counts()


# In[13]:


train_df.drop(train_df[train_df['gender']=='Other'].index, inplace=True)
test_df.drop(test_df[test_df['gender']=='Other'].index, inplace=True)


# ### Exploratory Data Analysis ( EDA )

# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

sns.set_style("darkgrid")


# In[16]:


plot_color = ['#008080', 'black']
sns.set_palette(['#008080', 'black'])


# In[17]:


# filter data for stroke and non stroke

no_stroke_data = train_df[train_df['stroke'] == 0]
stroke_data = train_df[train_df['stroke'] == 1]


# In[18]:


# univariate Analysis

stroke_count = train_df['stroke'].value_counts()


# In[19]:


stroke_count


# In[20]:


plt.pie(stroke_count, labels=['No','Yes'], autopct = '%1.1f%%', colors=['teal','orange']);
plt.title("Proportion of Stroke Cases")
plt.legend()


# In[21]:


# pip install -U kaleido


# In[22]:


# pip install --upgrade plotly


# In[23]:


# # Bivariate Analysis

# def custom_histogram(columns:str, titles:list)->go.Figure:
#     fig = sp.make_subplots(rows= 1, cols=2, subplot_titles = ['No Stroke', 'Yes Stroke'])
#     histogram_no_stroke = px.histogram(no_stroke_data, x=columns, color_discrete_sequence=['#008080'])
#     histogram_stroke = px.histogram(stroke_data, x=columns, color_discrete_sequence=['#008080'])
#     fig.add_trace(histogram_no_stroke['data'][0], row=1, col=1)
#     fig.add_trace(histogram_stroke['data'][0], row=1, col=2)
#     fig.update_layout(title = titles[0], title_x = 0.5)
#     fig.update_xaxes(title_text = titles[1])
#     fig.update_yaxes(title_text = 'Count')
#     return fig


# In[24]:


# fig = custom_histogram('avg_glucose_level', ['Average Glucose Level Distribution by Stroke', 'Average Glucose Level'])
# fig.show('svg')


# In[25]:


def custom_histogram(column: str, titles: list, no_stroke_data, stroke_data):
    # Set the figure size and layout (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(titles[0], fontsize=16)

    # First subplot: No Stroke
    sns.histplot(no_stroke_data[column], kde=False, color='#008080', ax=axes[0])
    axes[0].set_title("No Stroke")
    axes[0].set_xlabel(titles[1])
    axes[0].set_ylabel("Count")

    # Second subplot: Yes Stroke
    sns.histplot(stroke_data[column],  kde=False, color='#008080', ax=axes[1])
    axes[1].set_title("Yes Stroke")
    axes[1].set_xlabel(titles[1])
    axes[1].set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# In[26]:


custom_histogram(
    column='avg_glucose_level',
    titles=['Average Glucose Level Distribution by Stroke', 'Average Glucose Level'],
    no_stroke_data=no_stroke_data,
    stroke_data=stroke_data
)


# In[27]:


custom_histogram(
    column='age',
    titles=['age Distribution by Stroke', 'Age'],
    no_stroke_data=no_stroke_data,
    stroke_data=stroke_data
)


# In[28]:


stroke_data['bmi'].mean()


# In[29]:


no_stroke_data['bmi'].mean()


# In[30]:


sns.violinplot(y = 'bmi', x = 'stroke', data=train_df ) 
plt.title('BMI & Stroke Status')
plt.xlabel('Stroke ( 0 = No, 1 = Yes)')
plt.ylabel('BMI')
plt.show()


# In[31]:


train_df.columns


# In[32]:


x_values = ['smoking_status','hypertension', 'heart_disease','Residence_type','ever_married','gender']
fig, axis = plt.subplots(nrows = 3, ncols = 2, figsize = (12, 10))

for ax, x_value in zip(axis.flat, x_values):
    sns.countplot(data = train_df, x = x_value, hue='stroke', ax = ax)
    ax.set_title(f'{x_value.capitalize()} and Smoking status')

plt.tight_layout()
plt.show()


# In[33]:


# Data Preparation & Preprocessing
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import set_config

set_config(display= 'diagram')


# In[34]:


train_df.columns


# In[35]:


input_cols = list(train_df.columns)[1:-1]


# In[36]:


target_col = 'stroke'


# In[37]:


train_inputs = train_df[input_cols].copy()
train_target = train_df[target_col].copy()

test_inputs = test_df[input_cols].copy()


# In[38]:


numerical_cols = train_inputs.select_dtypes(include=np.number).columns.to_list()


# In[39]:


numerical_cols 


# In[40]:


categorical_cols = train_inputs.select_dtypes(include='object').columns.to_list()


# In[41]:


categorical_cols


# In[42]:


categorical_cols.remove('smoking_status')


# In[43]:


train_df['smoking_status'].unique()


# In[44]:


ordinal_enc = OrdinalEncoder(categories=[['Unknown','never smoked', 'formerly smoked','smokes']])


# In[45]:


onehot_enc = OneHotEncoder(handle_unknown='ignore')


# In[46]:


preprocessor = ColumnTransformer([
    ('Ordinal_Encode', ordinal_enc, ['smoking_status']),
    ('OneHot_Encode', onehot_enc, categorical_cols)
     ] )


# In[47]:


preprocessor.fit(train_inputs)


# In[48]:


encoded_cols = list(preprocessor.get_feature_names_out())


# In[49]:


encoded_cols


# In[50]:


train_inputs[encoded_cols] = preprocessor.transform(train_inputs)
test_inputs[encoded_cols] = preprocessor.transform(test_inputs)


# In[51]:


x_train = train_inputs[numerical_cols + encoded_cols]
y_train = train_df[target_col]

x_test = test_inputs[numerical_cols + encoded_cols]


# In[52]:


x_train.head()


# In[53]:


x_train.shape[1] == x_test.shape[1]


# In[54]:


#  split Dataset into training and validation set

from sklearn.model_selection import train_test_split


# In[55]:


X_train, X_val, y_train, y_val = train_test_split( x_train, y_train, test_size=.2, random_state = 42)


# In[56]:


X_train.shape


# In[57]:


pip install lightgbm


# In[58]:


pip install xgboost


# In[59]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# In[60]:


models = [
    (LogisticRegression(), 'LogisticRegression'),
    (RandomForestClassifier(), 'RandomForestClassifier'),
    (XGBClassifier(), 'XGBClassifier'),
    (LGBMClassifier(verbose = 1), 'LGBMClassifier')
     ]


# In[61]:


def evalue_model(model, X_train, y_train, X_val, y_val):
    y_pred = model.predict(X_val)
    confusion_mat = confusion_matrix(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    return confusion_mat, accuracy, precision, recall, f1


# In[62]:


confusion_mats = {}
def metric_df(model, model_name):
    confusion_mat, *df = evalue_model(model, X_train, y_train, X_val, y_val)
    confusion_mats[model_name] = confusion_mat
    df = [df]
    model_metrics = pd.DataFrame(data = df, columns=['Accuracy','Precision','Recall','F1'])
    model_metrics.insert(0, 'Model', model_name)

    return model_metrics


# In[63]:


predictions = []
for model, model_name in models:
    model.fit(X_train, y_train)
    model_metrics = metric_df(model, model_name)
    predictions.append(model_metrics)


# In[64]:


predictions


# In[65]:


oversample = SMOTE()
X_train_ns, y_train_ns = oversample.fit_resample(X_train, y_train)


# In[66]:


X_train_ns.shape


# In[67]:


lr_with_smote = LogisticRegression()
lr_with_smote.fit(X_train_ns, y_train_ns)
lr_with_smote_mat = metric_df(lr_with_smote,'Logistic With SMOTE')
predictions.append(lr_with_smote_mat)


# In[68]:


for model_name, confusion_mat in confusion_mats.items():
    print(f'Confusion Matrix for{model_name}')
    print(confusion_mat)
    print()


# In[69]:


# voting classifier . Hard and Soft
from sklearn.ensemble import VotingClassifier


# In[70]:


estimator = []
estimator.append(('LogisticRegression', LogisticRegression())),
estimator.append(('RandomForestClassifier', RandomForestClassifier())),
estimator.append(('XGBClassifier', XGBClassifier())),
estimator.append(('LGBMClassifier', LGBMClassifier(verbose = 1)))


# In[71]:


VC_hard = VotingClassifier(estimators=estimator, voting='hard')
VC_hard.fit(X_train, y_train)
VC_hard_mat = metric_df(VC_hard, 'VC_hard')
predictions.append(VC_hard_mat)


# In[72]:


VC_hard_mat


# In[73]:


VC_soft = VotingClassifier(estimators=estimator, voting='soft')
VC_soft.fit(X_train, y_train)
VC_soft_mat = metric_df(VC_soft, 'VC_soft')
predictions.append(VC_soft_mat)


# In[74]:


VC_soft_mat


# In[75]:


prediction_base = pd.concat(predictions, ignore_index=True, sort = False)
prediction_base = prediction_base.sort_values(by=['Recall'], ascending=False).style.hide(axis = 'index')


# In[76]:


prediction_base


# In[77]:


models_to_plot = [
    'LogisticRegression',
    'RandomForestClassifier',
    'XGBClassifier',
    'Logistic With SMOTE'
]


# In[78]:


fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(9,7))
for ax, model_name in zip(axes.ravel(), models_to_plot):
    confusion_mat = confusion_mats[model_name]
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', ax = ax)
    ax.set_title(f'{model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.subplots_adjust(left=0, right=0.9, top= 0.9, bottom=0)
plt.show()


# In[79]:


# AUC-Roc & precision-recall curve
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import PredictionErrorDisplay


# In[80]:


X_train = train_inputs[numerical_cols + encoded_cols]
y_train = train_df[target_col]

X_test = test_inputs[numerical_cols + encoded_cols]


# In[81]:


preds = lr_with_smote.predict(X_test)


# In[82]:


preds


# In[83]:


sub_df = preds


# In[84]:


pd.DataFrame(sub_df).to_csv("submission.csv", index=None)


# In[85]:


sub_df = pd.read_csv("submission.csv")


# In[86]:


sub_df.head()


# In[87]:


# prediction on single input
sub_df['stroke'] = preds


# In[88]:


sub_df


# In[89]:


train_df.columns


# In[90]:


single_input = {
    'gender':'Male',
    'age': 33,
    'hypertension':0,
    'heart_disease':0,
    'ever_married':'Yes',
    'work_type':'Private',
    'Residence_type':'Urban',
    'avg_glucose_level': 79.53,
    'bmi': 31.10,
    'smoking_status':'formerly smoked'
}


# In[91]:


input_df = pd.DataFrame([single_input])


# In[92]:


input_df[encoded_cols] = preprocessor.transform(input_df)


# In[93]:


X = input_df[numerical_cols + encoded_cols]


# In[94]:


lr_with_smote.predict(X)


# In[95]:


import joblib


# In[96]:


model = {
    'encoded_cols':encoded_cols,
    'numerical_cols':numerical_cols,
    'preprocessor':preprocessor,
    'model':lr_with_smote
}


# In[97]:


joblib.dump(model, 'model.joblib')


# In[ ]:





# In[ ]:




