import os
import math

import numpy as np
import tensorflow as tf
# from matplotlib import pyplot as plt


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
        # print(evaluate_data)
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


def cross_validation_estimate(data, train_steps, evaluate_steps, block_number):
    estimator = return_kernel_estimator()
    split_data = np.array_split(data, block_number)

    test_data = split_data[0]

    # print(test_data)

    split_data = np.array_split(np.vstack(np.delete(split_data, 0, 0)), block_number - 1)

    sum_evaluate_loss = 0
    sum_evaluate_accuracy = 0

    sum_test_loss = 0
    sum_test_accuracy = 0

    for i in range(0, block_number - 1):
        evaluate_data = split_data[i]

        # print(evaluate_data)

        train_data = np.vstack(np.delete(split_data, i, 0))

        ###
        input_fn_train, input_fn_evaluate, _ = return_input_fn(
            train_data, evaluate_data, None)
        estimator.fit(input_fn=input_fn_train, steps=train_steps)
        evaluate_results = estimator.evaluate(
            input_fn=input_fn_evaluate, steps=evaluate_steps)
        evaluate_accuracy = evaluate_results['accuracy']
        evaluate_loss = evaluate_results['loss']

        print('Block{0} evaluate accuracy is {1}'.format(i, evaluate_accuracy))
        print('Block{0} evaluate loss is {1}'.format(i, evaluate_loss))

        sum_evaluate_accuracy += evaluate_accuracy
        sum_evaluate_loss += evaluate_loss

        print()
        ###

        ###
        input_fn_train, input_fn_evaluate, _ = return_input_fn(
            train_data, test_data, None)
        estimator.fit(input_fn=input_fn_train, steps=train_steps)
        test_results = estimator.evaluate(
            input_fn=input_fn_evaluate, steps=evaluate_steps)
        test_accuracy = test_results['accuracy']
        test_loss = test_results['loss']

        print('Block{0} test accuracy is {1}'.format(i, test_accuracy))
        print('Block{0} test loss is {1}'.format(i, test_loss))

        sum_test_accuracy += test_accuracy
        sum_test_loss += test_loss

        print()
        ###

    ###
    evaluate_average_loss = sum_evaluate_loss / block_number
    evaluate_average_accuracy = sum_evaluate_accuracy / block_number
    print('Total evaluate loss average is {0}'.format(evaluate_average_loss))
    print('Total evaluate accuracy average is {0}'.format(evaluate_average_accuracy))
    print()
    ###

    ###
    test_average_loss = sum_test_loss / block_number
    test_average_accuracy = sum_test_accuracy / block_number
    print('Total test loss average is {0}'.format(test_average_loss))
    print('Total test accuracy average is {0}'.format(test_average_accuracy))
    print()
    ###

    return evaluate_average_loss, evaluate_average_accuracy, test_average_loss, test_average_accuracy

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



target_data_length_x = np.array([])
evaluate_average_loss_y = np.array([])
test_average_loss_y = np.array([])

division_number = 10
for i in range(1, division_number + 1):
    np.random.shuffle(data)
    split_data = np.array_split(data, division_number)
    target_data = np.vstack(split_data[:i])
    target_data_length = len(target_data)
    evaluate_average_loss, evaluate_average_accuracy, test_average_loss, test_average_accuracy = cross_validation_estimate(target_data, 2000, 100, 10)
    print('Data length is {0}'.format(target_data_length))
    print()
    print('Loss evaluate average is {0}'.format(evaluate_average_loss))
    print('Accuracy evaluate average is {0}'.format(evaluate_average_accuracy))
    print()
    print('Loss test average is {0}'.format(test_average_loss))
    print('Accuracy test average is {0}'.format(test_average_accuracy))

    print()
    target_data_length_x = np.append(target_data_length_x, target_data_length)
    evaluate_average_loss_y = np.append(evaluate_average_loss_y, evaluate_average_loss)
    test_average_loss_y = np.append(test_average_loss_y, test_average_loss)

x = target_data_length_x
y1 = evaluate_average_loss_y
y2 = test_average_loss_y

print(x)
print(y1)
print(y)

# plt.scatter(x , y1)
# plt.scatter(x , y2)
# plt.plot(x, np.poly1d(np.polyfit(x, y1, 3))(x), label='d=3')
# plt.plot(x, np.poly1d(np.polyfit(x, y2, 3))(x), label='d=3')
#
# plt.show()

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
