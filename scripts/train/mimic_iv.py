import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

patient_record_feature_description = {
    'target': tf.io.FixedLenFeature([], tf.int64),
    'x1': tf.io.FixedLenFeature([], tf.int64),
    'x2': tf.io.FixedLenFeature([], tf.float),
    'x3': tf.io.FixedLenFeature([], tf.float),
    'x4': tf.io.FixedLenFeature([], tf.float),
    'x5': tf.io.FixedLenFeature([], tf.float),
    'x6': tf.io.FixedLenFeature([], tf.float),
}

# Create model

model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[33]),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def patient_record_example(row_data):
    feature = {
        'target': _int64_feature(row_data[1]),
        'x1': _float_feature(row_data[3]),
        'x2': _float_feature(row_data[4]),
        'x3': _float_feature(row_data[5]),
        'x4': _float_feature(row_data[6]),
        'x5': _float_feature(row_data[7]),
        'x6': _float_feature(row_data[8]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def convert_csv_to_tfrecords():
    csv = pandas.read_csv("~/train.csv").values
    with tf.io.TFRecordWriter("~/train_csv.tfrecords") as writer:
        for row in csv:
            tf_example = patient_record_example(row)
            writer.write(tf_example.SerializeToString())

    csv = pandas.read_csv("~/test.csv").values
    with tf.io.TFRecordWriter("~/test_csv.tfrecords") as writer:
        for row in csv:
            tf_example = patient_record_example(row)
            writer.write(tf_example.SerializeToString())

def _parse_patient_record_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above
    example = tf.io.parse_single_example(example_proto, patient_record_feature_description)
    x = tf.stack([example['x1'],
                  example['x2'],
                  example['x3'],
                  example['x4'],
                  example['x5'],
                  example['x6']],axis=0)
    y = example['target']
    return x,y

def create_dataset(filepath):
    patient_record_dataset = tf.data.TFRecordDataset(filepath)
    parsed_patient_record_dataset = patient_record_dataset.map(_parse_patient_record_function)


def train(project_directory, model_directory, epochs, steps_per_epoch, validation_steps):
    filepath = '~/train_csv.tfrecords'
    train_data = create_dataset(filepath)

    filepath = '~/test_csv.tfrecords'
    test_data = create_dataset(filepath)

    # Add callbacks
    tensorboard_cb = tf.keras.callbacks.TensorBoard('~/logs')

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('~/weights.hdf5',
                                                       save_weights_only=True,
                                                       save_best_only=True,
                                                       save_freq='epoch',
                                                       monitor='val_loss',
                                                       mode='auto')

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                      min_delta=0.001,
                                                      restore_best_weights=True)

    history = model.fit(
        train_data,
        validation_data=(X_valid, y_valid),
        batch_size=512,
        epochs=100,
        callbacks=[early_stopping],
        verbose=0)


if __name__ == "__main__":
    history_df = pd.DataFrame(history.history)
    # Start the plot at epoch 5
    history_df.loc[5:, ['loss', 'val_loss']].plot()
    history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

    print(("Best Validation Loss: {:0.4f}" +\
           "\nBest Validation Accuracy: {:0.4f}")\
          .format(history_df['val_loss'].min(),
                  history_df['val_binary_accuracy'].max()))

    auc = tf.keras.metrics.AUC(
              num_thresholds=200, curve='ROC',
              summation_method='interpolation', name=None, dtype=None,
              thresholds=None, multi_label=False, label_weights=None)
    print(auc)

