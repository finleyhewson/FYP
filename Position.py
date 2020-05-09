import T265_Tracking_Camera as t265

from scipy.spatial.transform import Rotation as R
import numpy as np
import pymavlink
import time
from threading import Thread

class position:
    def __init__(self, pixhawkObj):
        self.t265 = t265.rs_t265()
        self.t265.open_connection()

        self.pixhawkObj = pixhawkObj

        self._pos = np.asarray([0,0,0], dtype=np.float)
        self._r = R.from_euler('xyz', [0,0,0])
        self._conf = 0

        self.running = True
        self.north_offset = None
        self.current_time_us = 0

    def setNorthOffset(self, north_offset):
        if north_offset is not None and self.north_offset is None:
            t265_yaw = self._r.as_euler('xyz')[0][2]
            north_offset -= t265_yaw 
            self.north_offset = R.from_euler('xyz', [0,0,north_offset])

    def __del__(self):
        self.t265.closeConnection()

    def update(self):
        return self._pos, self._r, self._conf

    def loop(self):
        while self.running:
            self._pos, self._r, self._conf, _ = self.t265.get_frame()
            #self.setNorthOffset( self.pixhawkObj.compass_heading )
            self.current_time_us = int(round(time.time() * 1000000))
            # Convert from FRD to NED coordinate system
            if self.north_offset is not None:
                self._pos = self.north_offset.apply(self._pos)
                self._r = self.north_offset * self._r
            
            if  self.pixhawkObj.enable_msg_vision_position_estimate:
                self.pixhawkObj.send_vision_position_estimate_message(self._pos, self._r, self.current_time_us)
                
            if self.pixhawkObj.enable_update_tracking_confidence_to_gcs:    
                self.pixhawkObj.send_tracking_confidence_to_gcs(self._conf)


            time.sleep(0.01)

    def close(self):
        self.running = False
