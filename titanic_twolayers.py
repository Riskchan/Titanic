import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.metrics as metrics
from sklearn import preprocessing

# データを読み込み、必要なものだけを取り出す
row_data = pd.read_csv('train.csv')
#data= row_data[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare']].dropna()
data= row_data[['Survived', 'Sex','Age','SibSp','Parch','Fare']].dropna()
cols = len(data.columns)

# Encode Sex and Embarked
labelEncoder = preprocessing.LabelEncoder()
data['Sex'] = labelEncoder.fit_transform(data['Sex'])

y = pd.get_dummies(data['Survived']).as_matrix()
x = data.drop('Survived',axis=1).values

# 変数の定義
node_num = 15   # 中間レイヤーのノード数
feature = tf.placeholder(tf.float32, [None, cols-1])
label = tf.placeholder(tf.float32, [None, 2])
w0 = tf.Variable(tf.truncated_normal([cols-1, node_num]))
b0 = tf.Variable(tf.zeros([node_num]))
w1 = tf.Variable(tf.truncated_normal([node_num,2]))
b1 = tf.Variable(tf.zeros([2]))

# モデルの定義
f0 = tf.matmul(feature, w0) + b0
p0 = tf.nn.sigmoid(f0)
f1 = tf.matmul(p0 , w1) + b1
p1 = tf.nn.softmax(f1)

# Loss function and optimization
loss = -tf.reduce_sum(label*tf.log(p1))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# 初期設定
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(50000):
    sess.run(optimizer, feed_dict={feature:x, label:y})  # 学習処理
    if i % 1000 == 0:
        loss_val = sess.run(loss, feed_dict={feature:x, label:y})
        predict = sess.run(p0 , feed_dict={feature:x , label:y})
        pre_label = np.argmax(predict, axis=1)
        acc_val = metrics.accuracy_score(pre_label, data['Survived'])             # 予測結果の評価
        print ('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))


