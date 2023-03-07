import json

import csv
import datetime

from ParkingSpot import ParkingSpot


class ParkingLot:
    parking_spots = []

    def __init__(self):
        parking_spots_data = self.import_json('final_kunratice_optimized.json')
        for parking_spot in parking_spots_data:
            self.parking_spots.append(
                ParkingSpot(parking_spot['id'],
                            False,  # not occupied
                            parking_spot['coordinates'],
                            parking_spot['handicapped'],
                            parking_spot['valid']))

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

    def add_parking_spot(self, spot_id, coordinates, handicapped, valid):
        self.parking_spots.append(ParkingSpot(spot_id,
                                              False,
                                              coordinates,
                                              handicapped,
                                              valid))

    def delete_parking_spot(self, spot_id):
        for i, spot in enumerate(self.parking_spots):
            if spot.get_id() == spot_id:
                del self.parking_spots[i]

    def dump_to_csv(self, image_order):
        time_now = datetime.datetime.now()
        all_count = self.get_all_parking_space_count()
        occupied_count = self.get_occupied_count()
        free_count = all_count - occupied_count
        occupied_list = list(filter(lambda parking_spot: parking_spot.is_occupied(), self.parking_spots))
        blabla = [o.get_id() for o in occupied_list]
        string_of_parking_spots = ' '.join(str(x) for x in blabla)
        csv_list = [image_order, time_now.strftime("%H:%M:%S"), all_count, occupied_count, free_count,
                    string_of_parking_spots]
        print(csv_list)
        with open('parking_statistics.csv', 'a') as f_object:
            writer_object = csv.writer(f_object, delimiter=';', quoting=csv.QUOTE_MINIMAL)
            writer_object.writerow(csv_list)
            f_object.close()
