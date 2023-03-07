import os
import cv2
import numpy as np
import schedule
import time
import datetime

from ParkingDetector import ParkingDetector
from ParkingLot import ParkingLot
from PlottingManager import PlottingManager
from YoloModel import YoloModel


class ObjectDetection:
    WIDTH = 1498
    HEIGHT = 720
    MINUTE_IN_SECONDS = 60

    def __init__(self, window_name, source):
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        self.finish = False
        self.window_name = window_name
        self.model = YoloModel('ultralytics/yolov5', 'yolov5m')
        self.plotting_manager = PlottingManager(self.WIDTH, self.HEIGHT)
        self.parking_lot = ParkingLot()
        self.parking_spots = self.parking_lot.get_parking_spots()
        self.parking_detector = ParkingDetector(self.WIDTH, self.HEIGHT)
        self.source = source
        self.cords = []

    def on_mouse(self, event, x, y, buttons, user_param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)

    def get_free_count(self):
        return self.parking_lot.get_all_parking_space_count() - self.parking_lot.get_occupied_count()

    def start(self):
        if self.source:
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        else:
            cap = cv2.VideoCapture(0)

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        last_recorded_time_score_frame = last_recorded_time_parking_spots = time.time()
        all_parking_spots_count = self.parking_lot.get_all_parking_space_count()
        free_parking_spots_count = occupied_parking_spots_count = 0
        start_time = time.time()
        image_order = 1
        create_folder()
        while True:
            # schedule.run_pending() todo delete
            curr_time = time.time()
            ret, frame = cap.read()

            frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
            frame[100:440, :] = 0  # black stripe

            # self.plotting_manager.preprocess_image(frame)

            if curr_time - last_recorded_time_score_frame >= 2.0:  # score frame every 2 seconds
                self.cords = self.model.score_frame(frame)
                last_recorded_time_score_frame = curr_time

            if curr_time - last_recorded_time_parking_spots >= 3.0:  # find IOU every 3 seconds
                self.parking_detector.detect_parking(self.cords, self.parking_spots)
                occupied_parking_spots_count = self.parking_lot.get_occupied_count()
                free_parking_spots_count = all_parking_spots_count - occupied_parking_spots_count
                last_recorded_time_parking_spots = curr_time

            if time.time() - start_time > self.MINUTE_IN_SECONDS:  # save image and csv every 60 seconds
                time_now = datetime.datetime.now()
                frame = self.plot_information(frame, occupied_parking_spots_count, free_parking_spots_count)
                cv2.imwrite("saved/frame_%d_%s.jpg" % (image_order, str(time_now.strftime("%H_%M_%S"))), frame)
                self.parking_lot.dump_to_csv(image_order)
                start_time = curr_time
                image_order = image_order + 1

            frame = self.plot_information(frame, occupied_parking_spots_count, free_parking_spots_count)

            cv2.imshow(self.window_name, frame)
            c = cv2.waitKey(1)
            if c == 27:
                self.finish = True
                break

        cap.release()
        cv2.destroyAllWindows()

    def plot_information(self, frame, occupied_parking_spots_count, free_parking_spots_count):
        frame = self.plotting_manager.plot_car_boxes(self.cords, frame)
        frame = self.plotting_manager.plot_parking_spots(self.parking_spots, frame)
        frame = self.plotting_manager.plot_statistics(occupied_parking_spots_count, free_parking_spots_count, frame)
        return frame

def create_folder():
    exist = os.path.exists('saved')
    if not exist:
        # Create a new directory because it does not exist
        os.makedirs('saved')