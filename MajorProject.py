import RPi.GPIO as GPIO
import drivers
import time
import cv2
import imutils
import pytesseract as pt
from time import sleep
import numpy as np
from picamera import PiCamera
from rpi_lcd import LCD

GPIO.setmode(GPIO.BOARD)
GPIO.setup(8, GPIO.IN)
GPIO.setup(12, GPIO.OUT)
servo1 = GPIO.PWM(12, 50)


camera = PiCamera()

lcd = LCD()


def cam():
    camera.start_preview()
    sleep(0.5)
    camera.capture('/home/pi/image.jpg')
    sleep(0.5)
    camera.stop_preview()


def plate_detect():
    resized_img = cv2.resize(img, (480, 360))
    cv2.imshow('Resized_img', resized_img)
    cv2.waitKey(0)

    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    blur_img = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imshow('blur_img', blur_img)
    cv2.waitKey(0)

    edged = cv2.Canny(blur_img, 30, 200)  # Perform Edge detection
    cv2.imshow('CannyEdge_img', edged)
    cv2.waitKey(0)

    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]
    screenCnt = None

    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
        print("No contour detected")

    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(resized_img, [screenCnt], -1, (0, 255, 0), 2)

    cv2.imshow('Cropped', resized_img)
    cv2.waitKey(0)

    # Masking the part other than the number plate
    mask_image = np.zeros(blur_img.shape, np.uint8)

    new_image = cv2.drawContours(mask_image, [screenCnt], 0, 255, -1, )

    new_image_1 = cv2.bitwise_and(resized_img, resized_img, mask=mask_image)

    # Now crop
    (x, y) = np.where(mask_image == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped = gray[topx:bottomx + 2, topy:bottomy + 2]
    cv2.imshow('crop', cropped)
    cv2.waitKey(0)
    return cropped


def num_detect(cropped):
    img1 = cv2.resize(cropped, (170, 38))
    cv2.imshow('crop', cropped)
    cv2.waitKey(0)
    ret, img2 = cv2.threshold(img1, 136, 245, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilation = cv2.dilate(img2, kernel, iterations=1)

    img3 = dilation.copy()
    cv2.imshow('Dilated_img', img3)
    cv2.waitKey(0)

    # print(pytesseract.image_to_string(img))
    # height, width, number of channels in image
    height, width = img3.shape

    boxes = pt.image_to_boxes(img3, config=" -c tessedit_create_boxfile=1", lang='nep')
    # print(pytesseract.image_to_string(img2, lang='nep'))

    num = " "

    for b in boxes.splitlines():
        b = b.split(' ')
        print(b)
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(img3, (x, height - y), (w, height - h), (0, 0, 255), 1)
        ratio = float((h - y) / (w - x))
        print(ratio)

    # cv2.imshow('boxes1',img3)
    # cv2.waitKey(0)
        # if height to width ratio is less than 1.5 skip
        if 0.5 < ratio < 2.55:
            cv2.rectangle(img3, (x, height - y), (w, height - h), (0, 0, 255), 1)
            num += b[0]

    cv2.imshow('boxes2', img3)
    cv2.waitKey(0)

    return num


def compare(plate_num):
    db = ['बा.२१च ७७८,\n', 'बा.१४च २8६४\n', 'बा१८च २५७८\n', ' बा.२झ १७७\n', ' बा.१ज ३४५७\n']
    taxis = [plate_num]
    print(taxis)
    for taxi in taxis:
        if taxi in db:
            lcd.text("Open the Gate", 1)
            sleep(7)
            lcd.clear()

            servo1.start(0)

            # Define variable duty
            duty = 2

            # Loop for duty values from 2 to 12 (0 to 180 degrees)
            while duty <= 8:
                servo1.ChangeDutyCycle(duty)
                time.sleep(1)
                duty = duty + 2

            # Wait a couple of seconds
            time.sleep(10)

            # turn back to 0 degrees
            print("Turning back to 0 degrees")
            servo1.ChangeDutyCycle(2)
            time.sleep(0.5)
            servo1.ChangeDutyCycle(0)

            # Clean things up at the end
            servo1.stop()
        else:
            lcd.text("No Entry", 1)
            lcd.clear()


while True:
    val1 = GPIO.input(8)
    print('value of ir in: ', val1)
    if val1 == 0:
        sleep(5)
        cam()
        img = cv2.imread('/home/pi/image.jpg')
        image = plate_detect()
        plate_num = num_detect(image)
        compare(plate_num)
        sleep(15)
