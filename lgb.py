import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
import datetime
import warnings
warnings.filterwarnings('ignore')
start = datetime.datetime.now()

train = pd.read_csv('train_processed1.csv')
# test = pd.read_csv('test_processed1.csv')
train['category_id_2'] = train['category_id_2'].astype(np.float32)
X_train, X_test, Y_train, Y_test = train_test_split(train['ITEM_NAME'], train['category_id_2'], random_state = 0, test_size=0.2, shuffle=True)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_counts = count_vect.transform(X_test)

X_train_tfidf = X_train_tfidf.astype('float32')
X_test_counts = X_test_counts.astype('float32')
Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')

clf_gbm=lgbm.LGBMClassifier(boosting_type='gbdt', objective='multiclass', learning_rate=0.02, num_leaves=60, max_depth=4,
                            n_estimators=2000, silent=True)

clf_gbm.fit(X_train_tfidf, Y_train, eval_set=[(X_train_tfidf, Y_train), (X_test_counts, Y_test)], early_stopping_rounds=50, verbose=10)      #x_train和y_train 是numpy或pandas数据类型即可
pre = clf_gbm.predict(X_test_counts)


clf_gbm.booster_.save_model('lgb_model1.txt', num_iteration=clf_gbm.best_iteration_)
print(accuracy_score(Y_test, pre))
# bst = lgbm.Booster(model_file='lgb_model.txt')
end = datetime.datetime.now()
print(end-start)
# train_X_1 = count_vect.transform(train['ITEM_NAME']).astype('float32')
# train['label'] = clf_gbm.predict(train_X_1)
# train.to_csv('train_lll.csv',index=False)