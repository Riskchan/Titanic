import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing


# Parameters
learning_rate = 0.01 # 学習率 高いとcostの収束が早まる
training_epochs = 100 # 学習全体をこのエポック数で区切り、区切りごとにcostを表示する
batch_size = 100     # 学習1回ごと( sess.run()ごと )に訓練データをいくつ利用するか
display_step = 1     # 1なら毎エポックごとにcostを表示
train_size = 800     # 全データの中でいくつ訓練データに回すか
step_size = 1000     # 何ステップ学習するか

# Network Parameters
n_hidden_1 = 64      # 隠れ層1のユニットの数
n_hidden_2 = 64      # 隠れ層2のユニットの数
n_input = 8          # 与える変数の数
n_classes = 2        # 分類するクラスの数 今回は生き残ったか否かなので2

df = pd.read_csv('train.csv')
labelEncoder = preprocessing.LabelEncoder()
df['Sex'] = labelEncoder.fit_transform(df['Sex'])
df['Cabin'] = labelEncoder.fit_transform(df['Cabin'].fillna('0'))
df['Embarked'] = labelEncoder.fit_transform(df['Embarked'].fillna('0'))

x_np = np.array(df[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']].fillna(0))
survived = preprocessing.LabelEncoder().fit_transform(df['Survived']).reshape(-1,1)
y_np = preprocessing.OneHotEncoder().fit_transform(survived).toarray()

[x_train, x_test] = np.vsplit(x_np, [train_size]) # 入力データを訓練データとテストデータに分ける
[y_train, y_test] = np.vsplit(y_np, [train_size]) # ラベルを訓練データをテストデータに分ける

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = multilayer_perceptron(x, weights, biases)
#cost = tf.reduce_mean(tf.square(pred - y))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(pred, y))
pred = tf.nn.softmax(pred)
cost = -tf.reduce_sum(y * tf.log(pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.

        # Loop over step_size
        for i in range(step_size):
            # 訓練データから batch_size で指定した数をランダムに取得
            ind = np.random.choice(batch_size, batch_size)
            x_train_batch = x_train[ind]
            y_train_batch = y_train[ind]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: x_train_batch, y: y_train_batch})
            # Compute average loss
            avg_cost += c / step_size
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: x_test, y: y_test}))
