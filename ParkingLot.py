import json

import numpy as np

from ParkingSpot import ParkingSpot


class ParkingLot:
    parking_spots = []

    def __init__(self):
        parking_spots_data = self.import_json('parking_lot.json')
        for parking_spot in parking_spots_data:
            self.parking_spots.append(
                ParkingSpot(parking_spot['id'],
                            False,  # not occupied
                            parking_spot['coordinates'],
                            parking_spot['handicapped']))

    def import_json(self, name):
        f = open(name)
        data = json.load(f)
        return data

    def get_parking_spots(self):
        return self.parking_spots

    def get_all_parking_space_count(self):
        return len(self.parking_spots)

    def get_occupied_count(self):
        occupied_list = list(filter(lambda parking_spot: parking_spot.is_occupied(), self.parking_spots))
        return len(occupied_list)

    def get_parking_spot(self, spot_id):
        for spot in self.parking_spots:
            if spot.get_id() == spot_id:
                return spot

    def add_parking_spot(self, spot_id, coordinates, handicapped):
        self.parking_spots.append(ParkingSpot(spot_id,
                                              False,
                                              coordinates,
                                              handicapped))

    def delete_parking_spot(self, spot_id):
        for i, spot in enumerate(self.parking_spots):
            if spot.get_id() == spot_id:
                del self.parking_spots[i]
