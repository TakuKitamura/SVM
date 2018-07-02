import tensorflow as tf
def input_fn():
    return {
      'example_id': tf.constant(['1', '2', '3']),
      'price': tf.constant([[0.6], [0.8], [0.3]]),
      'sq_footage': tf.constant([[900.0], [700.0], [600.0]]),
      'country': tf.SparseTensor(
          values=['IT', 'US', 'GB'],
          indices=[[0, 0], [1, 3], [2, 1]],
          dense_shape=[3, 5]),
      'weights': tf.constant([[3.0], [1.0], [1.0]])
    },tf.constant([[1], [0], [1]])

price = tf.contrib.layers.real_valued_column('price')
sq_footage_bucket = tf.contrib.layers.bucketized_column(
    tf.contrib.layers.real_valued_column('sq_footage'),
    boundaries=[650.0, 800.0])
country = tf.contrib.layers.sparse_column_with_hash_bucket(
    'country', hash_bucket_size=5)
sq_footage_country = tf.contrib.layers.crossed_column(
    [sq_footage_bucket, country], hash_bucket_size=10)
svm_classifier = tf.contrib.learn.SVM(
    feature_columns=[price, sq_footage_bucket, country, sq_footage_country],
    example_id_column='example_id',
    weight_column_name='weights',
    l1_regularization=0.1,
    l2_regularization=1.0)

svm_classifier.fit(input_fn=input_fn, steps=30)
