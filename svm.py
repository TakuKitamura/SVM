import os
import math

import numpy as np
import tensorflow as tf
from sklearn import preprocessing as pr

from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from flask import Flask, jsonify, abort, make_response, request
import datetime


def return_kernel_estimator(data):
    optimizer = tf.train.FtrlOptimizer(
        # learning_rate大きくすると過学習
        learning_rate=120.0, l1_regularization_strength=10, l2_regularization_strength=10)

    kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
        input_dim=data.shape[1]-1, output_dim=1, stddev=5, name='rffm')

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

        print(predict_data)

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
    estimator = return_kernel_estimator(data)
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

api = Flask(__name__)


def show_learnig_graph(data, division_number):
    target_data_length_x = np.array([])
    evaluate_average_loss_y = np.array([])
    test_average_loss_y = np.array([])


    print(data)
    # features_number = data.shape[1] - 1
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
    print(y2)

    plt.scatter(x , y1)
    plt.scatter(x , y2)
    plt.plot(x, np.poly1d(np.polyfit(x, y1, 3))(x), label='d=3')
    plt.plot(x, np.poly1d(np.polyfit(x, y2, 3))(x), label='d=3')

    plt.show()

def predict(analysis_data, predict_data, train_steps, evaluate_steps):
    np.random.shuffle(analysis_data)
    data_length = len(analysis_data)
    train_data_rate = 0.7

    if train_data_rate > 0 and train_data_rate < 1:
        split_index = math.floor(data_length * train_data_rate)
        train_data = analysis_data[:split_index]
        evaluate_data = analysis_data[split_index:]

        if len(train_data) == 0 or len(evaluate_data) == 0:
            raise ValueError("トレイニング用のデータ割合が不正です。")
    else:
        raise ValueError("トレイニング用のデータ割合が不正です。")

    input_fn_train, input_fn_evaluate, input_fn_predict = return_input_fn(
        train_data, evaluate_data, predict_data)

    estimator = return_kernel_estimator(analysis_data)
    estimator.fit(input_fn=input_fn_train, steps=train_steps)

    test_results = estimator.evaluate(
        input_fn=input_fn_evaluate, steps=evaluate_steps)

    test_accuracy = test_results['accuracy']
    test_loss = test_results['loss']

    print('Accuracy is {0}'.format(test_accuracy))
    print('Loss is {0}'.format(test_loss))

    predict_results = list(estimator.predict(input_fn=input_fn_predict))
    print(predict_results)
    logits = predict_results[0]["logits"][0]
    probabilities = predict_results[0]["probabilities"][0]
    classes = predict_results[0]["classes"]

    print('Classes is {0}'.format(classes))

    print('Logits is {0}'.format(logits))

    with tf.Session():
        print('Delay probabilities is {0}'.format(probabilities))

    @api.route('/predictKoseiLineDelay/<int:choiceTimeID>', methods=['GET'])
    def get_user(choiceTimeID):
        choiceTimeID
        responseText = ""
        if choiceTimeID == 1:
            responseText += "今日の朝の湖西線が、"
        elif choiceTimeID == 2:
            responseText += "今日の昼の湖西線が、"
        elif choiceTimeID == 3:
            responseText += "今日の夕方の湖西線が、"
        elif choiceTimeID == 4:
            responseText += "明日の朝の湖西線が、"
        elif choiceTimeID == 5:
            responseText += "明日の昼の湖西線が、"
        elif choiceTimeID == 6:
            responseText += "明日の夕方の湖西線が、"
        else:
            responseText = "不具合が発生しております。復旧まで居間しばらくお待ち下さい。"
            result = {"responseText": responseText}
            return make_response(jsonify(result))

        today = datetime.date.today()
        predict_data = np.array([np.average(analysis_data[np.where(analysis_data[:, 1] == today.month)], axis = 0)], dtype = 'float32')[:,0:-1]
        print(predict_data)
        _, _, input_fn_predict = return_input_fn(None, None, predict_data)

        predict_results = list(estimator.predict(input_fn=input_fn_predict))
        print(predict_results)
        delay_probabilities = predict_results[0]["probabilities"][1]

        if delay_probabilities * 100 < 20:
            responseText += "遅延する可能性はかなり低いです。"
        elif delay_probabilities * 100 < 40:
            responseText += "遅延する可能性は低いです。"
        elif delay_probabilities * 100 < 60:
            responseText += "遅延する可能性が有ります。"
        elif delay_probabilities * 100 <= 100:
            responseText += "遅延する可能性が高いです。"
        else:
            responseText = "不具合が発生しております。復旧まで居間しばらくお待ち下さい。"
            result = {"responseText": responseText}
            return make_response(jsonify(result))

        result = {"responseText": responseText}
        return make_response(jsonify(result))

    api.run(host='localhost', port=3000)


# Warning非表示
# 参考: https://qiita.com/KEINOS/items/4c66eeda4347f8c13abb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TODO tf.contrib.learnは非推薦なので修正する。
# https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/learn/README.md
# deprecated関数を使用していることによるWarningを非表示
tf.logging.set_verbosity(tf.logging.ERROR)

# np.set_printoptions(threshold=5000)

data = np.array(
    np.loadtxt("/Users/kitamurataku/work/svm/new_data.csv", delimiter=",", skiprows=1), "float64")


"""
### 横軸:月、縦軸:遅延回数の棒グラフ
notDelay = data[:, 1][np.where(data[:, -1] == 0)]
delay = data[:, 1][np.where(data[:, -1] == 1)]

plt.figure(1)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)

### 横軸:月、縦軸:遅延回数の棒グラフ(月前半)
month = np.arange(12) + 1
delayFlagCount = np.zeros(12, dtype=int)
for i in month:
    delayFlagCount[i - 1] = np.sum((data[:, 1] == i) & (data[:, -1] == 1) & (data[:, 2] <= 15))
# print(np.sum(delayFlagCount))

plt.figure(2)
plt.bar(month, delayFlagCount)

### 横軸:月、縦軸:遅延回数の棒グラフ(月後半)
month = np.arange(12) + 1
delayFlagCount = np.zeros(12, dtype=int)
for i in month:
    delayFlagCount[i - 1] = np.sum((data[:, 1] == i) & (data[:, -1] == 1) & (data[:, 2] > 15))
print(np.sum(delayFlagCount))

plt.figure(3)
plt.bar(month, delayFlagCount)



###降水量の合計
# bottom = np.array(data[:, 6][np.where(data[:, 6] < 150)])
# print(bottom)
# print(data[:, -1] == 0)
# plt.bar(bottom, data[:, -1][np.where(data[:, 6] < 150)] == 0, color='r', alpha=0.1)
# plt.bar(bottom, data[:, -1][np.where(data[:, 6] < 150)] == 1, color='b', alpha=0.1)
# plt.show()

###今津降水量の合計
notDelay = data[:, 3][np.where(data[:, -1] == 0)]
delay = data[:, 3][np.where(data[:, -1] == 1)]
plt.figure(4)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)



###南小松降水量の合計
notDelay = data[:, 4][np.where(data[:, -1] == 0)]
delay = data[:, 4][np.where(data[:, -1] == 1)]

plt.figure(5)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)



###大津降水量の合計
notDelay = data[:, 5][np.where(data[:, -1] == 0)]
delay = data[:, 5][np.where(data[:, -1] == 1)]
plt.figure(6)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)


###今津1時間降水量の最大
notDelay = data[:, 6][np.where(data[:, -1] == 0)]
delay = data[:, 6][np.where(data[:, -1] == 1)]
plt.figure(7)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###南小松1時間降水量の最大
notDelay = data[:, 7][np.where(data[:, -1] == 0)]
delay = data[:, 7][np.where(data[:, -1] == 1)]
plt.figure(8)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###大津1時間降水量の最大
notDelay = data[:, 8][np.where(data[:, -1] == 0)]
delay = data[:, 8][np.where(data[:, -1] == 1)]
plt.figure(9)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()


###平均風速
# bottom = np.array(data[:, 8])
# print(bottom)
# print(data[:, -1] == 0)
# plt.bar(bottom, data[:, -1] == 0, color='r', alpha=0.1)
# plt.bar(bottom, data[:, -1] == 1, color='b', alpha=0.1)
# plt.show()

###今津平均風速
notDelay = data[:, 9][np.where(data[:, -1] == 0)]
delay = data[:, 9][np.where(data[:, -1] == 1)]
plt.figure(10)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###南小松平均風速
notDelay = data[:, 10][np.where(data[:, -1] == 0)]
delay = data[:, 10][np.where(data[:, -1] == 1)]
plt.figure(11)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###大津平均風速
notDelay = data[:, 11][np.where(data[:, -1] == 0)]
delay = data[:, 11][np.where(data[:, -1] == 1)]
plt.figure(12)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###今津日照時間
notDelay = data[:, 12][np.where(data[:, -1] == 0)]
delay = data[:, 12][np.where(data[:, -1] == 1)]
plt.figure(13)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###南小松日照時間
notDelay = data[:, 13][np.where(data[:, -1] == 0)]
delay = data[:, 13][np.where(data[:, -1] == 1)]
plt.figure(14)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###大津日照時間
notDelay = data[:, 14][np.where(data[:, -1] == 0)]
delay = data[:, 14][np.where(data[:, -1] == 1)]
plt.figure(15)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###今津最深積雪
notDelay = data[:, 15][np.where(data[:, -1] == 0)]
delay = data[:, 15][np.where(data[:, -1] == 1)]
plt.figure(16)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###今津降雪量合計
notDelay = data[:, 16][np.where(data[:, -1] == 0)]
delay = data[:, 16][np.where(data[:, -1] == 1)]
plt.figure(17)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###今津最大風速
notDelay = data[:, 17][np.where(data[:, -1] == 0)]
delay = data[:, 17][np.where(data[:, -1] == 1)]
plt.figure(18)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###今津最大風速方角
notDelay = data[:, 18][np.where(data[:, -1] == 0)]
delay = data[:, 18][np.where(data[:, -1] == 1)]
plt.figure(19)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###今津最大瞬間風速
notDelay = data[:, 19][np.where(data[:, -1] == 0)]
delay = data[:, 19][np.where(data[:, -1] == 1)]
plt.figure(20)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###今津最大瞬間風速方角
notDelay = data[:, 20][np.where(data[:, -1] == 0)]
delay = data[:, 20][np.where(data[:, -1] == 1)]
plt.figure(21)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###今津最多風向
notDelay = data[:, 21][np.where(data[:, -1] == 0)]
delay = data[:, 21][np.where(data[:, -1] == 1)]
plt.figure(22)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###南小松最大風速
notDelay = data[:, 22][np.where(data[:, -1] == 0)]
delay = data[:, 22][np.where(data[:, -1] == 1)]
plt.figure(23)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###南小松最大風速
notDelay = data[:, 23][np.where(data[:, -1] == 0)]
delay = data[:, 23][np.where(data[:, -1] == 1)]
plt.figure(24)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###南小松最大瞬間風速方角
notDelay = data[:, 24][np.where(data[:, -1] == 0)]
delay = data[:, 24][np.where(data[:, -1] == 1)]
plt.figure(25)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###南小松最大瞬間風速方角
notDelay = data[:, 25][np.where(data[:, -1] == 0)]
delay = data[:, 25][np.where(data[:, -1] == 1)]
plt.figure(26)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###南小松最多風向
notDelay = data[:, 26][np.where(data[:, -1] == 0)]
delay = data[:, 26][np.where(data[:, -1] == 1)]
plt.figure(27)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###大津最大風速
notDelay = data[:, 27][np.where(data[:, -1] == 0)]
delay = data[:, 27][np.where(data[:, -1] == 1)]
plt.figure(28)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###大津最大風速方角
notDelay = data[:, 28][np.where(data[:, -1] == 0)]
delay = data[:, 28][np.where(data[:, -1] == 1)]
plt.figure(29)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###大津最大瞬間風速
notDelay = data[:, 29][np.where(data[:, -1] == 0)]
delay = data[:, 29][np.where(data[:, -1] == 1)]
plt.figure(30)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###大津最大瞬間風速方角
notDelay = data[:, 30][np.where(data[:, -1] == 0)]
delay = data[:, 30][np.where(data[:, -1] == 1)]
plt.figure(31)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###大津最多風向
notDelay = data[:, 31][np.where(data[:, -1] == 0)]
delay = data[:, 31][np.where(data[:, -1] == 1)]
plt.figure(32)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###今津平均気温
notDelay = data[:, 32][np.where(data[:, -1] == 0)]
delay = data[:, 32][np.where(data[:, -1] == 1)]
plt.figure(33)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###今津最高気温
notDelay = data[:, 33][np.where(data[:, -1] == 0)]
delay = data[:, 33][np.where(data[:, -1] == 1)]
plt.figure(34)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###今津最低気温
notDelay = data[:, 34][np.where(data[:, -1] == 0)]
delay = data[:, 34][np.where(data[:, -1] == 1)]
plt.figure(35)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###南小松平均気温
notDelay = data[:, 35][np.where(data[:, -1] == 0)]
delay = data[:, 35][np.where(data[:, -1] == 1)]
plt.figure(36)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###南小松最高気温
notDelay = data[:, 36][np.where(data[:, -1] == 0)]
delay = data[:, 36][np.where(data[:, -1] == 1)]
plt.figure(37)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###南小松最低気温
notDelay = data[:, 37][np.where(data[:, -1] == 0)]
delay = data[:, 37][np.where(data[:, -1] == 1)]
plt.figure(38)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###大津平均気温
notDelay = data[:, 38][np.where(data[:, -1] == 0)]
delay = data[:, 38][np.where(data[:, -1] == 1)]
plt.figure(39)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###大津最高気温
notDelay = data[:, 39][np.where(data[:, -1] == 0)]
delay = data[:, 39][np.where(data[:, -1] == 1)]
plt.figure(40)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###大津最低気温
notDelay = data[:, 40][np.where(data[:, -1] == 0)]
delay = data[:, 40][np.where(data[:, -1] == 1)]
plt.figure(41)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()


###今津10分間降水量の最大
notDelay = data[:, 41][np.where(data[:, -1] == 0)]
delay = data[:, 41][np.where(data[:, -1] == 1)]
plt.figure(42)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###南小松10分間降水量の最大
notDelay = data[:, 42][np.where(data[:, -1] == 0)]
delay = data[:, 42][np.where(data[:, -1] == 1)]
plt.figure(43)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

###大津10分間降水量の最大
notDelay = data[:, 43][np.where(data[:, -1] == 0)]
delay = data[:, 43][np.where(data[:, -1] == 1)]
plt.figure(44)
plt.hist([notDelay, delay], bins=50, color=['red', 'blue'], label=['x1', 'x2'], histtype='bar', stacked=True)
# plt.show()

# plt.show()

"""

data = np.c_[
    data[:, 0], # 年
    data[:, 1], # 月
    data[:, 2], # 日付
    data[:, 3], # 今津降水量の合計(mm)
    data[:, 4], # 南小松降水量の合計(mm)
    data[:, 5], # 大津降水量の合計(mm)
    data[:, 6], # 今津1時間降水量の最大(mm)
    data[:, 7], # 南小松1時間降水量の最大(mm)
    data[:, 8], # 大津1時間降水量の最大(mm)
    data[:, 9], # 今津平均風速(m/s)
    data[:, 10],# 南小松平均風速(m/s)
    data[:, 11],# 大津平均風速(m/s)
    data[:, 12],# 今津日照時間(時間)
    data[:, 13],# 南小松日照時間(時間)
    data[:, 14],# 大津日照時間(時間)
    data[:, 15],# 今津最深積雪(cm)
    data[:, 16],# 今津降雪量合計(cm)
    data[:, 17],# 今津最大風速(m/s)
    data[:, 18],# 今津最大風速方角(m/s)
    data[:, 19],# 今津最大瞬間風速(m/s)
    data[:, 20],# 今津最大瞬間風速方角(m/s)
    data[:, 21],# 今津最多風向(16方位)
    data[:, 22],# 南小松最大風速(m/s)
    data[:, 23],# 南小松最大風速方角(m/s)
    data[:, 24],# 南小松最大瞬間風速(m/s)
    data[:, 25],# 南小松最大瞬間風速方角(m/s)
    data[:, 26],# 南小松最多風向(16方位)
    data[:, 27],# 大津最大風速(m/s)
    data[:, 28],# 大津最大風速方角(m/s)
    data[:, 29],# 大津最大瞬間風速(m/s)
    data[:, 30],# 大津最大瞬間風速方角(m/s)
    data[:, 31],# 大津最多風向(16方位)
    data[:, 32],# 今津平均気温(℃)
    data[:, 33],# 今津最高気温(℃)
    data[:, 34],# 今津最低気温(℃)
    data[:, 35],# 南小松平均気温(℃)
    data[:, 36],# 南小松最高気温(℃)
    data[:, 37],# 南小松最低気温(℃)
    data[:, 38],# 大津平均気温(℃)
    data[:, 39],# 大津最高気温(℃)
    data[:, 40],# 大津最低気温(℃)
    data[:, 41],# 今津10分間降水量の最大(mm)
    data[:, 42],# 南小松10分間降水量の最大(mm)
    data[:, 43],# 大津10分間降水量の最大(mm)
    data[:, 44],# 月の前半かフラグ(前半0)
    data[:, 45],# 遅延フラグ(遅延1)
]

print(data)


np.random.shuffle(data)
print(data.shape)

# data = pr.scale(data)

# unique, count = np.unique(data[:, -1], return_counts=True)
#
# little_label_number = unique[np.argmin(count)]
# many_label_number = unique[np.argmax(count)]
#
# # little_label_number = 1
# # many_label_number = 0
# # print(np.round(data[:, -1], 2), np.round(little_label_number, 2), np.round(many_label_number, 2))
# litte_data = data[np.where(data[:, -1] == little_label_number)]
# many_data = data[np.where(data[:, -1] == many_label_number)]
#
# data = np.r_[litte_data, many_data[0:int(len(litte_data))]]
#
# print(len(litte_data))
# print(len(many_data[0:int(len(litte_data))]))

# sm = SMOTE()
# features, labels = sm.fit_sample(data[:, 0:-1], data[:, -1])

# data = np.c_[features, labels]

# print(data[-1])

# print(features)
#
# print(labels)

# data = pr.scale(data)

# print(data)

# show_learnig_graph(data, 10)
predict(data, np.array([data[0][0:-1]]), 2000, 1)
