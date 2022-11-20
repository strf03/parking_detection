import numpy as np
from shapely.geometry import Polygon


class ParkingDetector:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def detectParking(self, cars_coordinates, parking_spots):
        for i, parking_spot in enumerate(parking_spots):
            parking_spot_poly = Polygon(parking_spot.get_coordinates())
            parking_spot.set_occupied(False)
            for cord in cars_coordinates:
                if cord[4] >= 0.2:  # todo treshold
                    x1, y1, x2, y2 = int(cord[0] * self.width), int(
                        cord[1] * self.height), int(cord[2] * self.width), int(
                        cord[3] * self.height)
                    car_poly = Polygon([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
                    polygon_intersection = parking_spot_poly.intersection(car_poly).area
                    polygon_union = parking_spot_poly.union(car_poly).area
                    IOU = round((polygon_intersection / polygon_union) * 100) # lepsi pocitani IOU https://github.com/sainimohit23/parking-space-detection-system/blob/master/detector.py
                    if IOU >= 25:
                        parking_spot.set_occupied(True)
