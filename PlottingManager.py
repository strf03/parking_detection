import datetime

import cv2
import numpy as np
from shapely.geometry import Polygon

# from ParkingLot.ParkingLot() import get_occupied_count

CAR_COLOR = (255, 0, 0)


class PlottingManager:

    THRESHOLD = 0.2
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def plot_car_boxes(self, cords, frame):
        for cord in cords:
            if cord[4] >= self.THRESHOLD:
                x1, y1, x2, y2 = int(cord[0] * self.width), int(
                    cord[1] * self.height), int(cord[2] * self.width), int(
                    cord[3] * self.height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), CAR_COLOR, 1)
        return frame

    def plot_parking_spots(self, parking_spots, frame):
        for parking_spot in parking_spots:
            cv2.polylines(frame, [np.array(parking_spot.get_coordinates())], True,
                          parking_spot.get_color(), 1)
            parking_spot_poly = Polygon(parking_spot.get_coordinates())
            centroid = parking_spot_poly.centroid
            # cv2.circle(frame, (round(centroid.x), round(centroid.y)), 5, parking_spot.get_color(), -1)
            cv2.putText(frame, str(parking_spot.get_id()), (round(centroid.x)-10, round(centroid.y)), cv2.FONT_HERSHEY_SIMPLEX,
                        1, parking_spot.get_color(), 3)
        return frame

    def preprocess_image(self, frame):
        pts1 = np.float32([[0, 430], [1498, 430],
                           [0, 720], [1498, 720]])
        pts2 = np.float32([[0, 0], [1498, 0],
                           [0, 720], [1498, 720]])

        matrix, status = cv2.findHomography(pts1, pts2)
        frame = cv2.warpPerspective(frame, matrix, (1498, 720))

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        frame = cv2.warpPerspective(frame, matrix, (1498, 720))

    def plot_statistics(self, occupied_count, free_count, frame):
        all_count = occupied_count + free_count  # todo dict + cycle
        cv2.putText(frame, 'All parking spaces:' + str(all_count), (215, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 1, 2)

        cv2.putText(frame, 'Occupied spaces:' + str(occupied_count), (215, 240), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 1, 2)
        cv2.putText(frame, 'Free spaces:' + str(free_count), (215, 320), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 1, 2)
        cv2.putText(frame, 'Time:' + datetime.datetime.now().strftime("%H:%M:%S"), (215, 400), cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255), 1, 2)
        return frame
