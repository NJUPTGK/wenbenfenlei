import pandas as pd
import pkuseg
import datetime
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')
print("程序开始执行")
start = datetime.datetime.now()

train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')
train['class_l1'] = train['TYPE'].apply(lambda x:x.split(sep='--')[0])
train['class_l2'] = train['TYPE'].apply(lambda x:x.split(sep='--')[1])
train['class_l3'] = train['TYPE'].apply(lambda x:x.split(sep='--')[2])
# TYPE = list(set(train['TYPE']))

col = ['ITEM_NAME','class_l1','class_l2','class_l3']
train = train[col]
train.columns = ['ITEM_NAME', 'class_l1', 'class_l2', 'class_l3']
train['category_id_1'] = train['class_l1'].factorize()[0]
category_id_df_1 = train[['class_l1', 'category_id_1']].drop_duplicates().sort_values('category_id_1')
category_to_id_1 = dict(category_id_df_1.values)
id_to_category_1 = dict(category_id_df_1[['category_id_1','class_l1']].values)
train['category_id_2'] = train['class_l2'].factorize()[0]
category_id_df_2 = train[['class_l2', 'category_id_2']].drop_duplicates().sort_values('category_id_2')
category_to_id_2 = dict(category_id_df_2.values)
id_to_category_2 = dict(category_id_df_2[['category_id_2','class_l2']].values)
train['category_id_3'] = train['class_l3'].factorize()[0]
category_id_df_3 = train[['class_l3', 'category_id_3']].drop_duplicates().sort_values('category_id_3')
category_to_id_3 = dict(category_id_df_3.values)
id_to_category_3 = dict(category_id_df_3[['category_id_3','class_l3']].values)

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False
def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False
def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False

def format_str(content):
    content_str = ''
    for i in content:
        if is_chinese(i) or is_number(i) or is_alphabet(i):
            content_str = content_str + i
    return content_str


train['ITEM_NAME1'] = train['ITEM_NAME'].apply(format_str)
# test['ITEM_NAME1'] = test['ITEM_NAME'].apply(format_str)

seg = pkuseg.pkuseg()
train['ITEM_NAME1'] = train['ITEM_NAME1'].apply(lambda x: ' '.join(seg.cut(x)))
# test['ITEM_NAME1'] = test['ITEM_NAME1'].apply(lambda x: ' '.join(seg.cut(x)))
print("分词完成")
X = train['ITEM_NAME']
y1 = train['category_id_1']
y2 = train['category_id_2']
y3 = train['category_id_3']


count_vect_1 = CountVectorizer()
X_train_counts_1 = count_vect_1.fit_transform(X)
tfidf_transformer_1 = TfidfTransformer()
X_train_tfidf_1 = tfidf_transformer_1.fit_transform(X_train_counts_1)
clf_1 = LinearSVC()
clf_1 = clf_1.fit(X_train_tfidf_1, y1)
joblib.dump(clf_1, '''train_model_1.m''')

count_vect_2 = CountVectorizer()
X_train_counts_2 = count_vect_2.fit_transform(X)
tfidf_transformer_2 = TfidfTransformer()
X_train_tfidf_2 = tfidf_transformer_2.fit_transform(X_train_counts_2)
clf_2 = LinearSVC()
clf_2 = clf_2.fit(X_train_tfidf_2, y2)
joblib.dump(clf_2, '''train_model_2.m''')

count_vect_3 = CountVectorizer()
X_train_counts_3 = count_vect_3.fit_transform(X)
tfidf_transformer_3 = TfidfTransformer()
X_train_tfidf_3 = tfidf_transformer_3.fit_transform(X_train_counts_3)
clf_3 = LinearSVC()
clf_3 = clf_3.fit(X_train_tfidf_3, y3)
joblib.dump(clf_3, '''train_model_3.m''')
def Yuce_1(sentences):
    label = clf_1.predict(count_vect_1.transform([str(sentences)]))
    return label[0]

def Yuce_2(sentences):
    label = clf_2.predict(count_vect_2.transform([str(sentences)]))
    return label[0]

def Yuce_3(sentences):
    label = clf_3.predict(count_vect_3.transform([str(sentences)]))
    return label[0]

print("开始预测")
train['category_id_1'] = train['ITEM_NAME1'].apply(Yuce_1)
train['category_id_2'] = train['ITEM_NAME1'].apply(Yuce_2)
train['category_id_3'] = train['ITEM_NAME1'].apply(Yuce_3)

print("预测完成")
train = train[['ITEM_NAME', 'category_id_1', 'category_id_2', 'category_id_3']]

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

train['TYPE'] = train['class_l1'].str.cat([train['class_l2'], train['class_l3']], sep='--')
train = train[['ITEM_NAME', 'TYPE']]
train.to_csv('train1.csv', index=False)

end = datetime.datetime.now()
print(end-start)