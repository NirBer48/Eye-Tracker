import numpy as np
import pygame as pg
import pygamebg
import cv2
import time
import pickle as pkl

cap = cv2.VideoCapture(0)
target_image = pg.image.load("target.png")
img = cv2.imread("target.png")
face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("model/haarcascade_eye.xml")
white = (255, 255, 255)
green = (0, 255, 0)
red = (255, 0, 0)
blue = (0, 0, 128)


def test_loop():
    pg.init()

    window = pg.display.set_mode((1920, 1080))
    clock = pg.time.Clock()
    font = pg.font.Font('freesansbold.ttf', 256)
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

    data = []
    labels = []
    pos = []

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

        (mouse_x, mouse_y) = pg.mouse.get_pos()
        # show the image centered
        (x_pos, y_pos) = (mouse_x - target_image.get_width() / 2, mouse_y - target_image.get_height() / 2)
        # print(f"mouse X: {mouse_x} mouse Y: {mouse_y}")
        mouse_location = location_font.render(f"({mouse_x},{mouse_y})", True, 0)

        location_rect = mouse_location.get_rect()
        location_rect.center = (window.get_width() // 2, window.get_height() - 80)

        img = capture()
        # img = cv2.imread("target.png")

        camera_label_text = "Connected" if img is not None else "No camera"
        camera_info = info_font.render(camera_label_text, True, red)
        current_time = time.time()

        if (current_time - last_capture) > 0.15:
            if img is not None:
                eyes = detect_eyes(img)

                if (eyes is not None):
                    # cv2.imwrite(f"{current_time}.png", eyes)
                    # writer_csv.writerow([img, [mouse_x, mouse_y]])
                    data.append(eyes)
                    labels.append([mouse_x, mouse_y])
                    pos.append(get_positions(img))
                    last_capture = current_time

        info_rect = camera_info.get_rect()
        info_rect.center = (window.get_width() // 2, window.get_height() // 2)

        # draw the scene
        window.blit(target_image, (x_pos, y_pos))
        window.blit(camera_info, info_rect)
        window.blit(mouse_location, location_rect)

        # update the display
        pg.display.flip()

    with open("correction_train.pkl", "wb") as f:
        pkl.dump([data, labels, pos], f)

    f.close()
    cap.release()
    pg.quit()
    exit()


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


def get_positions(img):
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


if "__main__" == __name__:
    test_loop()
