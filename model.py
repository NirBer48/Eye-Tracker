from random import shuffle
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import pickle as pkl

with open("data/train.pkl", "rb") as f:
    train_x, train_y = pkl.load(f)

with open("data/train2.pkl", "rb") as f:
    train_x2, train_y2 = pkl.load(f)
    
with open("data/train3.pkl", "rb") as f:
    train_x3, train_y3 = pkl.load(f)

with open("data/train4.pkl", "rb") as f:
    train_x4, train_y4 = pkl.load(f)

with open("data/train5.pkl", "rb") as f:
    train_x5, train_y5 = pkl.load(f)

train_x = np.concatenate((train_x, train_x2, train_x3, train_x4, train_x5))
train_y = np.concatenate((train_y, train_y2, train_y3, train_y4, train_y5))

images_length = len(train_x)
print(images_length)

# cv2.imshow(str(train_y[80]), train_x[80])
# cv2.waitKey(0)

def normalize_img(image, label):
  return tf.cast(image, tf.float32) / 255., [label[0] / 1920, label[1] / 1080]

ds_train = tf.data.Dataset.from_tensor_slices((train_x[:int(images_length*0.8)], train_y[:int(images_length*0.8)]))
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(4000, reshuffle_each_iteration=True)
ds_train = ds_train.batch(10)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


ds_test = tf.data.Dataset.from_tensor_slices((train_x[int(images_length*0.8):], train_y[int(images_length*0.8):]))
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(4)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(45,90,1)),
        tf.keras.layers.Conv2D(64, kernel_size=(4, 2), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2)
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="mse",
    metrics='accuracy',
)

model.fit(
    ds_train,
    epochs=15,
    validation_data=ds_test,
)

test_loss, test_acc = model.evaluate(ds_test)
print('\nTest accuracy: {}'.format(test_acc))

model.save('saved_model/my_model.h5')

# label=train_y[400]
# image, _= normalize_img(train_x[400])

ds_numpy = list(ds_test.as_numpy_iterator())

print("#######################")
print(f"predict: {model.predict(ds_numpy[5][0])}")
print(f"actual: {ds_numpy[5][1]}")
