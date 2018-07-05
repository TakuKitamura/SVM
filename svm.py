import os
import math

import numpy as np
import tensorflow as tf
from sklearn import datasets

# Warning非表示
# 参考: https://qiita.com/KEINOS/items/4c66eeda4347f8c13abb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TODO tf.contrib.learnは非推薦なので修正する。
# https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/learn/README.md
# deprecated関数を使用していることによるWarningを非表示
tf.logging.set_verbosity(tf.logging.ERROR)

sess = tf.Session()

# iris = datasets.load_iris()


# data = iris.data
data = np.array(np.loadtxt("/Users/kitamurataku/work/SVM/tmp.csv", delimiter=","))
data_length = len(data)
# print(data)
# data_type_flag = iris.target

# 各データの分類 (0: 'setosa', 1: 'versicolor', 2: 'virginica')
# decision_flag = np.array([[1] if y == 0 else [0] for y in data_type_flag])

# analysis_data = np.c_[data, decision_flag]


analysis_data = data
# np.random.shuffle(analysis_data)
np.random.shuffle(analysis_data)

# [[7.6 3.  6.6 2.1 0. ]
# [6.  3.  4.8 1.8 0. ]
# [4.4 2.9 1.4 0.2 1. ]
# [6.7 3.  5.  1.7 0. ]
# [5.9 3.  5.1 1.8 0. ]]

train_data_rate = 0.8

if train_data_rate > 0 and train_data_rate < 1:
    split_index = math.floor(data_length * train_data_rate)
    train_data = analysis_data[:split_index]
    evaluate_data = analysis_data[split_index:]

    if len(train_data) == 0 or len(evaluate_data) == 0:
        raise ValueError("トレイニング用のデータ割合が不正です。")
else:
    raise ValueError("トレイニング用のデータ割合が不正です。")

# column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
column_names = ['year', 'month', 'day', 'week_day', 'open', 'high', 'low', 'close', 'volume']

def return_input_fn(train_data, evaluate_data, predict_data, column_names, steps):

    train_data_length = len(train_data)
    evaluate_data_length = len(evaluate_data)
    column_names_length = len(column_names)

    def input_fn_train():  # returns x, y
        # x = {'example_id': tf.constant([str(i) for i in range(train_data_length)])}
        # for i in range(column_names_length):
        #     x[column_names[i]] = tf.convert_to_tensor(train_data[:, i])

        # x = {'feature': tf.constant(train_data[:, 0:-1])}
        x = {}
        x['feature'] = tf.constant(train_data[:, 0:-1], "float32")

        y = tf.constant(train_data[:, -1], "int64")

        return x, y

    def input_fn_evaluate():  # returns x, y
        # x = {'example_id': tf.constant([str(i) for i in range(train_data_length + 1 ,train_data_length + column_names_length)])}
        # for i in range(column_names_length):
        #     x[column_names[i]] = tf.convert_to_tensor(evaluate_data[:, i])

        x = {}
        print(evaluate_data[:, 0:-1])
        x['feature'] = tf.constant(evaluate_data[:, 0:-1], "float32")
        y = tf.constant(evaluate_data[:, -1], "int64")

        return x, y

    def input_fn_predict():
        # x = {'example_id': tf.constant([str(train_data_length + column_names_length + 1)])}
        # for i in range(column_names_length):
        #     x[column_names[i]] = tf.convert_to_tensor(predict_data[i])

        x = {}
        print(predict_data)
        x['feature'] = tf.constant(predict_data, "float32")
        y = None

        return x, y

    # if mode == "train":
    #     return input_fn_train
    # elif mode == "eval":
    #     return input_fn_eval
    # elif mode == "predict":
    #     return input_fn_predict
    # else:
    #     raise ValueError("modeが不適切です。")

    # feature_columns =[tf.contrib.layers.real_valued_column(name) for name in column_names]

    optimizer = tf.train.FtrlOptimizer(learning_rate=0.1)

    kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
        input_dim=9, output_dim=1)

    kernel_mappers = {tf.contrib.layers.real_valued_column('feature'): [kernel_mapper]}

    # for column in feature_columns:
    #     kernel_mappers[column] = [kernel_mapper]

    # print(kernel_mappers)
    estimator = tf.contrib.kernel_methods.KernelLinearClassifier(
        feature_columns=[], optimizer=optimizer, kernel_mappers=kernel_mappers)


    # estimator = tf.contrib.learn.SVM(
    #     example_id_column='example_id',
    #     feature_columns=feature_columns,
    # )

    # print(123)
    estimator.fit(input_fn=input_fn_train, steps=2000)

    # print(456)
    print(estimator.evaluate(input_fn=input_fn_evaluate, steps=1))
    # print(789)
    results = list(estimator.predict(input_fn=input_fn_predict))
    print(results)
    logits = results[0]["logits"]

    with tf.Session() as sess:
        probabilities = 1 / (1 + np.exp(-logits))
        print(probabilities)
# column_names = ['year', 'month', 'day', 'week_day', 'open', 'high', 'low', 'close', 'volume']

predict_data = np.array([[2016, 1, 12, 1, 6800, 6864, 6755, 6755, 12126300]])

# for i in range(0, 100):
return_input_fn(train_data, evaluate_data, predict_data, column_names, 1)
