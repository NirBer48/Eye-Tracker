import numpy as np
import pygame as pg
import pygamebg
import cv2
import time
from tensorflow import keras

cap = cv2.VideoCapture(0)
target_image = pg.image.load("target.png")
face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("model/haarcascade_eye.xml")
model = keras.models.load_model('saved_model/my_model.h5')
white = (255, 255, 255)
green = (0, 255, 0)
red = (255, 0, 0)
blue = (0, 0, 128)


def test_loop():
    pg.init()

    window = pg.display.set_mode((1920, 1080))
    clock = pg.time.Clock()
    font = pg.font.Font('freesansbold.ttf', 256)
    x_pos, y_pos = 960, 640
    # main application loop

    for i in range(1, 4):
        window.fill(white)
        text = font.render(str(4 - i), True, red)
        textRect = text.get_rect()
        textRect.center = (window.get_width() // 2, window.get_height() // 2)
        window.blit(text, textRect)
        pg.display.flip()
        time.sleep(1)

    info_font = pg.font.Font('freesansbold.ttf', 64)
    location_font = pg.font.Font('freesansbold.ttf', 20)

    # data_csv = open('train_data.csv', 'w', newline='')
    # writer_csv = csv.writer(data_csv)

    last_capture = time.time()
    run = True

    while run:

        # limit frames per second
        clock.tick(100)

        # event loop
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_q:
                    run = False

        # clear the display
        window.fill(white)

        img = capture()
        # img = cv2.imread("target.png")

        camera_label_text = "Connected" if img is not None else "No camera"
        camera_info = info_font.render(camera_label_text, True, red)
        current_time = time.time()

        if (current_time - last_capture) > 0.1:
            if img is not None:
                eyes = detect_eyes(img)

                if (eyes is not None):
                    x_pos, y_pos = get_pos(eyes)
                    # correction = linear_model.predict([get_correction_pos(img)])
                    # window.blit(target_image, np.add([correction[0][0] * 1920, correction[0][1] * 1080], (x_pos, y_pos)))
                    window.blit(target_image, (x_pos, y_pos))
                    last_capture = current_time

        info_rect = camera_info.get_rect()
        info_rect.center = (window.get_width() // 2, window.get_height() // 2)

        # draw the scene
        
        window.blit(camera_info, info_rect)
        window.blit(target_image, (x_pos, y_pos))

        # update the display
        pg.display.flip()

    cap.release()
    pg.quit()
    exit()


def get_correction_pos(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        x, y, w, h = faces[0]
        x_pos = x + (w / 2)
        y_pos = y + (h / 2)
        z_pos = (w * h) / (1920 * 1080)
        print(x_pos, y_pos, z_pos)

        return x_pos, y_pos, z_pos
    except:
        return None


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


def detect_eyes(img):
    # save the image(i) in the same directory
    # img = capture()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        x, y, w, h = faces[0]
        # img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        left_index = 1 if eyes[0][0] < eyes[1][0] else 0

        left_x, left_y, left_w, left_h = eyes[left_index]
        right_x, right_y, right_w, right_h = eyes[1 - left_index]

        left_eye = roi_gray[left_y:left_y+left_h, left_x:left_x+left_w]
        right_eye = roi_gray[right_y:right_y+right_h, right_x:right_x+right_w]

        left_eye = cv2.resize(left_eye, (45,45))
        right_eye = cv2.resize(right_eye, (45,45))

        return np.concatenate((right_eye, left_eye), axis=1)

    except:
        pass
    return None


def capture():
    ret, frame = cap.read()

    if ret:
        return frame

    return None


if "__main__" == __name__:
    test_loop()

