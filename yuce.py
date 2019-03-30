import pandas as pd
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')
start = datetime.datetime.now()

train = pd.read_csv('train_processed1.csv')
# test = pd.read_csv('test_processed3.csv')

f1 = open('id_to_category_1.txt','r')
a1 = f1.read()
id_to_category_1 = eval(a1)
f1.close()
f2 = open('id_to_category_2.txt','r')
a2 = f2.read()
id_to_category_2 = eval(a2)
f2.close()
f3 = open('id_to_category_3.txt','r')
a3 = f3.read()
id_to_category_3 = eval(a3)
f3.close()

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train['ITEM_NAME'])
X_test_counts = count_vect.transform(train['ITEM_NAME'])




def Yuce_3(sentences):
    label = clf_3.predict(count_vect.transform([str(sentences)]))
    return label[0]
clf_1 = joblib.load('train_model_1.m')
clf_2 = joblib.load('train_model_2.m')
clf_3 = joblib.load('train_model_3.m')
# train['category_id_3'] = clf_3.predict(X_test_counts)
train['category_id_1'] = clf_1.predict(X_test_counts)
train['category_id_2'] = clf_2.predict(X_test_counts)
train['category_id_3'] = train['ITEM_NAME'].apply(Yuce_3)
def zhuanhuan_1(num):
    name = id_to_category_1[num]
    return name

def zhuanhuan_2(num):
    name = id_to_category_2[num]
    return name

def zhuanhuan_3(num):
    name = id_to_category_3[num]
    return name

train['class_l1'] = train['category_id_1'].apply(zhuanhuan_1)
train['class_l2'] = train['category_id_2'].apply(zhuanhuan_2)
train['class_l3'] = train['category_id_3'].apply(zhuanhuan_3)
train['TYPE1'] = train['class_l1'].str.cat([train['class_l2'],train['class_l3']],sep='--')
train = train[['TYPE1']]
train.to_csv('train_processed5.csv',index=False)
end = datetime.datetime.now()
print(end-start)