import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.metrics as metrics
from sklearn import preprocessing

# データを読み込み、必要なものだけを取り出す
row_data = pd.read_csv('train.csv')
data= row_data[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare']].dropna()
cols = len(data.columns)

# Encode Sex and Embarked
labelEncoder = preprocessing.LabelEncoder()
data['Sex'] = labelEncoder.fit_transform(data['Sex'])

y = data['Survived'].values.reshape(-1, 1)
x = data.drop('Survived',axis=1).values

# 変数の定義
feature = tf.placeholder(tf.float32, [None, cols-1])
label = tf.placeholder(tf.float32, [None, 1])
w0 = tf.Variable(tf.zeros([cols-1, 1]))
b0 = tf.Variable(tf.zeros([1]))

# モデルの定義
f0 = tf.matmul(feature, w0) + b0
p0 = tf.sigmoid(f0)

# Loss function and optimization
loss = -tf.reduce_sum(label*tf.log(p0) + (1-label)*tf.log(1-p0))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# 初期設定
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    sess.run(optimizer, feed_dict={feature:x, label:y})  # 学習処理
    if i % 100 == 0:
        loss_val = sess.run(loss, feed_dict={feature:x, label:y})
        predict = sess.run(p0 , feed_dict={feature:x , label:y})
        pre_label = [0 if x<0.5 else 1 for x in predict]
        acc_val = metrics.accuracy_score(pre_label, data['Survived'])             # 予測結果の評価
        print ('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))


