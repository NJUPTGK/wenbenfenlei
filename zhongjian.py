import pandas as pd
# df1 = pd.read_csv('test.csv')
# df2 = pd.read_csv('test_processed4.csv')
# result = pd.concat([df1, df2], axis=1, join='outer')
# result.to_csv('test_pre.csv',index=False)
df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('train_processed5.csv')
result = pd.concat([df1, df2], axis=1, join='outer')
result.to_csv('train1.csv',index=False)

