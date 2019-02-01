import pandas as pd
import jieba
import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
warnings.filterwarnings('ignore')
print("程序开始执行")
start = datetime.datetime.now()
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train['class_l1'] = train['TYPE'].apply(lambda x:x.split(sep='--')[0])
train['class_l2'] = train['TYPE'].apply(lambda x:x.split(sep='--')[1])
train['class_l3'] = train['TYPE'].apply(lambda x:x.split(sep='--')[2])
# train.to_csv('train_processed1.csv',index=False)
# train = pd.read_csv('train_processed1.csv')
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
# train.to_csv('train_processed2.csv', index=False)
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

# train = pd.read_csv('train_processed2.csv')
train['ITEM_NAME'] = train['ITEM_NAME'].apply(format_str)
test['ITEM_NAME1'] = test['ITEM_NAME'].apply(format_str)
# train.to_csv('train_processed3.csv',index=False)
# train = pd.read_csv('train_processed3.csv')
train['ITEM_NAME'] = train['ITEM_NAME'].apply(lambda x: ' '.join(jieba.cut(x)))
test['ITEM_NAME1'] = test['ITEM_NAME1'].apply(lambda x: ' '.join(jieba.cut(x)))
print("分词完成")
X = train['ITEM_NAME'].values
y1 = train['category_id_1'].values
y2 = train['category_id_2'].values
y3 = train['category_id_3'].values

train_X_1, test_x_1, train_y_1, test_y_1 = train_test_split(X, y1, test_size=0.1)
count_vect_1 = CountVectorizer()
X_train_counts_1 = count_vect_1.fit_transform(train_X_1)
tfidf_transformer_1 = TfidfTransformer()
X_train_tfidf_1 = tfidf_transformer_1.fit_transform(X_train_counts_1)
clf_1 = LinearSVC()
clf_1 = clf_1.fit(X_train_tfidf_1, train_y_1)

train_X_2, test_x_2, train_y_2, test_y_2 = train_test_split(X, y2, test_size=0.1)
count_vect_2 = CountVectorizer()
X_train_counts_2 = count_vect_2.fit_transform(train_X_2)
tfidf_transformer_2 = TfidfTransformer()
X_train_tfidf_2 = tfidf_transformer_2.fit_transform(X_train_counts_2)
clf_2 = LinearSVC()
clf_2 = clf_2.fit(X_train_tfidf_2, train_y_2)

train_X_3, test_x_3, train_y_3, test_y_3 = train_test_split(X, y3, test_size=0.1)
count_vect_3 = CountVectorizer()
X_train_counts_3 = count_vect_3.fit_transform(train_X_3)
tfidf_transformer_3 = TfidfTransformer()
X_train_tfidf_3 = tfidf_transformer_3.fit_transform(X_train_counts_3)
clf_3 = LinearSVC()
clf_3 = clf_3.fit(X_train_tfidf_3, train_y_3)

def Yuce_1(sentences):
    label = clf_1.predict(count_vect_1.transform([str(sentences)]))
    return label

def Yuce_2(sentences):
    label = clf_2.predict(count_vect_2.transform([str(sentences)]))
    return label

def Yuce_3(sentences):
    label = clf_3.predict(count_vect_3.transform([str(sentences)]))
    return label

print("开始预测")
test['category_id_1'] = test['ITEM_NAME1'].apply(Yuce_1)
test['category_id_2'] = test['ITEM_NAME1'].apply(Yuce_2)
test['category_id_3'] = test['ITEM_NAME1'].apply(Yuce_3)

print("预测完成")
test = test[['ITEM_NAME','category_id_1','category_id_2','category_id_3']]

def zhuanhuan_1(num):
    name = id_to_category_1[num[0]]
    return name

def zhuanhuan_2(num):
    name = id_to_category_2[num[0]]
    return name

def zhuanhuan_3(num):
    name = id_to_category_3[num[0]]
    return name

test['class_l1'] = test['category_id_1'].apply(zhuanhuan_1)
test['class_l2'] = test['category_id_2'].apply(zhuanhuan_2)
test['class_l3'] = test['category_id_3'].apply(zhuanhuan_3)

test['TYPE'] = test['class_l1'].str.cat([test['class_l2'],test['class_l3']],sep='--')
test = test[['ITEM_NAME','TYPE']]
test.to_csv('test_predict.csv',index=False)
end = datetime.datetime.now()
print(end-start)
