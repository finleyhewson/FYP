import pymavlink
import time
import numpy as np
from threading import Thread

class motion:
    def __init__(self, pixhawkObj):
        self.running = True
        self.pixhawkObj = pixhawkObj
        self.recalc_path = False
        return 
    def update(self, xpath, ypath, yawpath):
        self.xpath = xpath
        self.ypath = ypath
        self.yawpath = yawpath
    
    def loop(self):
        if self.running:
            if self.recalc_path == False:
                for self.ixpath, self.iypath, self.iyawpath in zip(self.xpath, self.ypath, self.yawpath):
                    if self.recalc_path == True:
                        break
                    else:
                        self.pixhawkObj.send_ned_position(self.ixpath, self.iypath, 0)
                        self.pixhawkObj.condition_yaw(self.iyawpath, relative = False)
                        time.sleep(0.1)

            else:
                self.pixhawkObj.send_ned_position(0, 0, 0)          
                print("recalculating path")
                time.sleep(0.1)

    def recalc(self, recalc_path):
        self.recalc_path = recalc_path

    def close(self):
        self.running = False


