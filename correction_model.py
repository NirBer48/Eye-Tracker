from random import shuffle
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = tf.keras.models.load_model('saved_model/my_model.h5')

def get_pos(eyes):
    eyes = eyes.reshape(1, 45, 90, 1)
    eyes = eyes / 255.
    pos = model.predict(eyes)
    (x_pos, y_pos) = pos[0][0] * 1920, pos[0][1] * 1080
    x_pos = max(0, x_pos)
    x_pos = min(1920, x_pos)
    y_pos = max(0, y_pos)
    y_pos = min(1080, y_pos)

    return x_pos, y_pos

with open("correction_train.pkl", "rb") as f:
    train_x, mouse_pos, person_pos = pkl.load(f)

for img_inx in range(len(train_x)):
    mouse_pos[img_inx] = np.subtract(mouse_pos[img_inx], get_pos(train_x[img_inx]))

images_length = len(person_pos)
print(images_length)

# cv2.imshow(str(train_y[80]), train_x[80])
# cv2.waitKey(0)
def normalize_positions(person_pos ,mouse_pos):
  return [person_pos[0] / 1920, person_pos[1] / 1080, person_pos[2]], [mouse_pos[0] / 1920, mouse_pos[1] / 1080]

for img_inx in range(len(train_x)):
    person_pos[img_inx], mouse_pos[img_inx] = normalize_positions(person_pos[img_inx], mouse_pos[img_inx])

ds_train = (person_pos[:int(images_length*0.8)], mouse_pos[:int(images_length*0.8)])
ds_test = (person_pos[int(images_length*0.8):], mouse_pos[int(images_length*0.8):])

linear_model = LinearRegression().fit(ds_train[0], ds_train[1])

predictions = linear_model.predict(ds_test[0])
print(mean_squared_error(ds_test[1] ,predictions))

# label=train_y[400]
# image, _= normalize_img(train_x[400])

# ds_numpy = list(ds_test.as_numpy_iterator())

# print("#######################")
# print(f"predict: {model.predict(ds_numpy[5][0])}")
# print(f"actual: {ds_numpy[5][1]}")
