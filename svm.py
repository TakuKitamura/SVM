import os
import math

import numpy as np
import tensorflow as tf


def return_kernel_estimator():
    optimizer = tf.train.FtrlOptimizer(learning_rate=1.2)

    kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
        input_dim=9, output_dim=1)

    kernel_mappers = {
        tf.contrib.layers.real_valued_column('feature'): [kernel_mapper]}

    estimator = tf.contrib.kernel_methods.KernelLinearClassifier(
         n_classes=2, optimizer=optimizer, kernel_mappers=kernel_mappers)

    return estimator


def return_input_fn(train_data, evaluate_data, predict_data):

    def input_fn_train():
        x = {}
        x['feature'] = tf.constant(train_data[:, 0:-1], "float32")

        y = tf.constant(train_data[:, -1], "int64")

        return x, y

    def input_fn_evaluate():

        x = {}
        x['feature'] = tf.constant(evaluate_data[:, 0:-1], "float32")
        y = tf.constant(evaluate_data[:, -1], "int64")

        return x, y

    def input_fn_predict():

        x = {}
        x['feature'] = tf.constant(predict_data, "float32")
        y = None

        return x, y

    return input_fn_train, input_fn_evaluate, input_fn_predict

    # estimator.fit(input_fn=input_fn_train, steps=2000)
    #
    # evaluate_results = estimator.evaluate(input_fn=input_fn_evaluate, steps=100)['loss']
    #
    # return evaluate_results

    # results = list(estimator.predict(input_fn=input_fn_predict))
    # print(results)
    # logits = results[0]["logits"]

    # with tf.Session() as sess:
    #     probabilities = 1 / (1 + np.exp(-logits))
    #     print(probabilities)


def cross_validation_estimate(data, train_steps, evaluate_steps):
    estimator = return_kernel_estimator()
    split_data = np.array_split(data, 10)
    sum_loss = 0
    sum_accuracy = 0
    for i in range(0, 10):
        evaluate_data = split_data[i]
        train_data = np.vstack(np.delete(split_data, i, 0))
        input_fn_train, input_fn_evaluate, _ = return_input_fn(
            train_data, evaluate_data, None)
        estimator.fit(input_fn=input_fn_train, steps=train_steps)
        evaluate_results = estimator.evaluate(
            input_fn=input_fn_evaluate, steps=evaluate_steps)
        accuracy = evaluate_results['accuracy']
        loss = evaluate_results['loss']

        print('Block{0} accuracy is {1}'.format(i, accuracy))
        print('Block{0} loss is {1}'.format(i, loss))

        sum_accuracy += accuracy
        sum_loss += loss

        print()

    average_loss = sum_loss / 10
    average_accuracy = sum_accuracy / 10
    print('Total loss average is {0}', average_loss)
    print('Total accuracy average is {0}', average_accuracy)

    return average_loss, average_accuracy

# Warning非表示
# 参考: https://qiita.com/KEINOS/items/4c66eeda4347f8c13abb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TODO tf.contrib.learnは非推薦なので修正する。
# https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/learn/README.md
# deprecated関数を使用していることによるWarningを非表示
tf.logging.set_verbosity(tf.logging.ERROR)

# np.set_printoptions(threshold=np.inf)

# sess = tf.Session()
#
data = np.array(
    np.loadtxt("/Users/kitamurataku/work/SVM/tmp.csv", delimiter=","), "int64")

np.random.shuffle(data)

cross_validation_estimate(data, 2000, 100)
#
# analysis_data = data
#
# np.random.shuffle(analysis_data)
#
# data_length = len(analysis_data)
#
# train_data_rate = 0.7
#
# if train_data_rate > 0 and train_data_rate < 1:
#     split_index = math.floor(data_length * train_data_rate)
#     train_data = analysis_data[:split_index]
#     evaluate_data = analysis_data[split_index:]
#
#     if len(train_data) == 0 or len(evaluate_data) == 0:
#         raise ValueError("トレイニング用のデータ割合が不正です。")
# else:
#     raise ValueError("トレイニング用のデータ割合が不正です。")


# predict_data = np.array([[2016, 1, 12, 1, 6800, 6864, 6755, 6755, 12126300]])
#
# return_input_fn(train_data, evaluate_data, predict_data, 1)
