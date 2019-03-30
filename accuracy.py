import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import datetime
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
warnings.filterwarnings('ignore')
print("程序开始执行")
start = datetime.datetime.now()

train = pd.read_csv('train_processed1.csv')

X_train, X_test, Y_train, Y_test = train_test_split(train['ITEM_NAME'], train['category_id_1'], random_state = 0, test_size=0.2, shuffle=True)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_counts = count_vect.transform(X_test)




clf = LinearSVC()
clf = clf.fit(X_train_tfidf, Y_train)
pre = clf.predict(X_test_counts)
print(accuracy_score(Y_test, pre))

end = datetime.datetime.now()
print(end-start)