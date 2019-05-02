import numpy
import urllib3
import urllib
import time
import cv2
from datetime import datetime, timedelta
import base64
import json
import requests
import os, os.path


class FeatureExtraction:
    def __init__(self):
        self.fpp = Facepp()
        self.frame_counter = 1
        self.start_time = self.now()
        self.delta = timedelta(seconds=1.2)
        self.block = False
        self.counter = len(
            [name for name in os.listdir("./imgs/") if os.path.isfile(name)]
        )

    def now(self):
        return datetime.now()

    def parse_frame(self, frame):
        # if self.now() - self.start_time > self.delta and not self.block:
        self.block = True
        self.counter += 1
        path = "./imgs/mmegg " + str(self.counter) + ".jpeg"
        cv2.imwrite(path, frame)

        try:
            data = self.fpp.parse_frame(path)
            print(data)
            return data

        except Exception as e:
            print(e)

        self.start_time = self.now()
        self.block = False


class Facepp:
    def __init__(self):
        self.http_url = "https://api-us.faceplusplus.com/facepp/v3/detect"
        self.key = "LX2oRHchELnqIswcEGQMCpMslc05lE7m"
        self.secret = "4sK29DKX4gJrJ-wv9l4RQyR0VlQQS9Mw"
        self.http = urllib3.PoolManager()

    def getBaseFormat(self):
        data_dict = {}

        data_dict["api_key"] = self.key
        data_dict["return_attributes"] = "gender,age,ethnicity"
        data_dict["api_secret"] = self.secret

        return data_dict

    def parse_frame(self, path):
        querystring = self.getBaseFormat()

        files = {
            "image_file": ("whatever.jpg", open(path, "rb")),
            "Content-Type": "image/jpeg",
        }

        try:
            # post data to server
            resp = requests.request(
                "POST", self.http_url, files=files, params=querystring
            )
            qrcont = resp.text
            jason = json.loads(qrcont)
            print(jason)
            return jason["faces"]
        except Exception as e:
            print(e)




