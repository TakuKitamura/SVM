# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

# Warning非表示
# 参考: https://qiita.com/KEINOS/items/4c66eeda4347f8c13abb
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.Session()

iris = datasets.load_iris()


# ex.
# [[5.1 0.2],
#  [4.9 0.2],
#  [4.7 0.2],
# ]
x_vals = np.array([[x[0], x[3]] for x in iris.data])

# ex.
# [1 -1 1 ]
y_vals = np.array([[1] if y == 0 else [0] for y in iris.target])

# print(x_vals, y_vals)


# 乱数を用いてトレーニングデータを八割準備
train_indices = \
 np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)

x_vals_train = x_vals[train_indices]
y_vals_train = y_vals[train_indices]

# x_vals_train = tf.constant(x_vals[train_indices], "float32")
# y_vals_train = tf.constant(y_vals[train_indices], "float32")

# print(x_vals_train, x_vals_train)

# /Users/kitamurataku/.pyenv/versions/3.6.5/lib/python3.6/site-packages/tensorflow/contrib/layers/python/layers/feature_column.py
# テストデータを二割準備
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_test = x_vals[test_indices]
y_vals_test = y_vals[test_indices]


real_feature_columnA = tf.contrib.layers.real_valued_column('A')

# real_feature_columnB = tf.contrib.layers.real_valued_column('B')

# real_feature_columnC = tf.contrib.layers.real_valued_column('C')


# sparse_feature_column = \
#  tf.contrib.layers.sparse_column_with_hash_bucket("y", hash_bucket_size=100)
# print(77777777)
estimator = tf.contrib.learn.SVM(
    example_id_column='example_id',
    feature_columns=[real_feature_columnA],
)

# print(888888888)

# abc = tf.constant(0, "int32",[120,1])
def input_fn_train():  # returns x, y
    # print(abc)
    # print(x_vals_train)
    # print(y_vals)
    # print(x_vals_train)
    print(x_vals_train)
    print(y_vals_train)

    return {
        'example_id': tf.constant([str(i) for i in range(120)]),
        'A': tf.convert_to_tensor(x_vals_train),
        # 'B': tf.convert_to_tensor(y_vals_train)
    }, tf.constant(y_vals_train)

def input_fn_eval():  # returns x, y
    # print(abc)
    # print(x_vals)
    # print(y_vals)
    return {
        'example_id': tf.constant([str(i) for i in range(121, 150)]),
        'A': tf.convert_to_tensor(x_vals_test),
        # 'B': tf.convert_to_tensor(y_vals_test)
    }, tf.constant(y_vals_test)


def predict_input_fn():
    return {
        'example_id': tf.constant(['151']),
        'A': tf.convert_to_tensor([[8, 0.25]]),
        # 'B': tf.convert_to_tensor([[0]]),
    }, None


# def input_fn_eval():  # returns x, y
#     x = x_vals_test
#     y = y_vals_test
#
#     return x, y

# print(type(input_fn_train()[0]))
# steps = 1000
# print(000000000)
print(123)
estimator.fit(input_fn=input_fn_train, steps=1000)
print(456)
# print(list(estimator.predict(input_fn=predict_input_fn)))

estimator.evaluate(input_fn=input_fn_eval, steps=1000)

print()

results = list(estimator.predict(input_fn=predict_input_fn))
logits = results[0]["logits"]

with tf.Session() as sess:
    probabilities = 1 / (1 + np.exp(-logits))
    print(probabilities)
# print(111111111)
# estimator.evaluate(input_fn=input_fn_train)
# print(222222222)
# p0 = list(estimator.predict(x=10))
# print(p0)
# print(next(estimator.predict(x=10)))
# print(3333333333)
