import cv2
import numpy as np
from shapely.geometry import Polygon

#from ParkingLot.ParkingLot() import get_occupied_count

CAR_COLOR = (255, 0, 0)


class PlottingManager:

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def plot_boxes(self, cords, frame):  # todo rename
        for cord in cords:
            if cord[4] >= 0.2:  # treshold 0.2 (20%) todo constant
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
            cv2.circle(frame, (round(centroid.x), round(centroid.y)), 5, parking_spot.get_color(), -1)
        return frame

    def plot_statistics(self, parking_spots, frame):
        pass
        #ParkingLot().get_all_parking_space_count()
