import numpy as np
import py
from scipy import interpolate
from scipy import io
import sys
import traceback
import copy
import time

class mapper:
    num_coordinate = 3

    xRange = [-8, 8]
    yRange = [-8, 8]
    zRange = [-0.5, 0.5]

    voxelSize = 0.02
    voxelMaxWeight = 2000
    voxelWeightDecay = 1

    xDivisions = int((xRange[1] - xRange[0]) / voxelSize)
    yDivisions = int((yRange[1] - yRange[0]) / voxelSize)
    zDivisions = int((zRange[1] - zRange[0]) / voxelSize)

    def __init__(self):
        self.xBins = np.linspace(self.xRange[0], self.xRange[1], self.xDivisions)
        self.yBins = np.linspace(self.yRange[0], self.yRange[1], self.yDivisions)
        self.zBins = np.linspace(self.zRange[0], self.zRange[1], self.zDivisions)

        self.grid = np.zeros((self.xDivisions, self.yDivisions, self.zDivisions), dtype=np.float32)

        self.interpFunc = interpolate.RegularGridInterpolator( (self.xBins, self.yBins, self.zBins),
                                                               self.grid, method = 'linear',
                                                               bounds_error = False,
                                                               fill_value = np.nan )

    # --------------------------------------------------------------------------
    # frame_to_global_points
    # param frame - (3,X,Y) matrix of coordinates from d435 camera
    # param pos - [x,y,z] offset cooridnates
    # param r - scipy local->global rotation object
    # return Null
    # --------------------------------------------------------------------------
    def local_to_global_points(self, local_points, pos, r):
        # Transform into global coordinate frame

        points_global = r.apply(local_points)
        points_global = np.add(points_global, pos)

        return points_global

    # --------------------------------------------------------------------------
    # updateMap
    # param pos - (N,3) list of points to add to the map
    # param rot -
    # return Null
    # --------------------------------------------------------------------------
    def update(self, points, pos, rot):
        # Add to map

        points = self.local_to_global_points(points, pos, rot)     
        self.updateMap(points, pos)
        self.interpFunc.values = self.grid


    def digitizePoints(self, points):

        xSort = np.digitize(points[:, 0], self.xBins) -1  #Facing Directly Forward from the camera
        ySort = np.digitize(points[:, 1], self.yBins) -1  #Direction to the right of the camera, facing away from it
        zSort = np.digitize(points[:, 2], self.zBins) -1  #Direction straight up from the camera

        return [xSort, ySort, zSort]

    # --------------------------------------------------------------------------
    # updateMap
    # param points - (N,3) list of points to qadd to the map
    # return Null
    # --------------------------------------------------------------------------
    def updateMap(self, points, pos):
        
        # Update map
        gridPoints = self.digitizePoints(points)
  
        np.add.at(self.grid, gridPoints, 1)

        # Decay map where map has not reached maxWeight
        self.grid = np.where(self.grid < self.voxelMaxWeight, 
                             self.grid - self.voxelWeightDecay, #If True 
                             self.grid) #If False

        # Keep all map values below voxelMaxWeight
        self.grid = np.clip(self.grid, a_min=0, a_max=self.voxelMaxWeight)

    # --------------------------------------------------------------------------
    # queryMap
    # param queryPoints - (N,3) list of points to query against map
    # return (N) list of risk for each point
    # --------------------------------------------------------------------------
    def queryMap(self, queryPoints):
        return self.interpFunc(queryPoints)

    def saveToMatlab(self, filename):
        io.savemat(filename, mdict=dict(map=self.grid), do_compression=False)



if __name__ == "__main__":
#def mainpc():
   # from modules.realsense 
    import T265_Tracking_Camera as t265
    import D435_Depth_Camera as d435
    import Telemetry as telemetry

    import cv2
    import base64
    import time
    import threading

    t265Obj = t265.rs_t265()
    d435Obj = d435.rs_d435(framerate=30, width=480, height=270)

    mapObj = mapper()

    with t265Obj, d435Obj:
        try:
            while True:
                t13 = time.perf_counter()
                # Get frames of data - points and global 6dof
                pos, r, conf, _ = t265Obj.get_frame()

                frame, rgbImg = d435Obj.getFrame()
                points = d435Obj.deproject_frame(frame)
                mapObj.update(points, pos, r)
                
                depth = cv2.applyColorMap(cv2.convertScaleAbs(frame, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow('frame', depth)
                cv2.waitKey(1)
                #print(conf)
                try:
                    
                    x = np.digitize(pos[0], mapObj.xBins) - 1
                    y = np.digitize(pos[1], mapObj.yBins) - 1
                    z = np.digitize(pos[2], mapObj.zBins) - 1
                    z2= np.digitize(pos[2], mapObj.zBins) - 2
                    z3= np.digitize(pos[2], mapObj.zBins) - 0

                    gridSlice1=copy.copy(mapObj.grid[:,:,z])
                    gridSlice2=copy.copy(mapObj.grid[:,:,z2])
                    gridSlice3=copy.copy(mapObj.grid[:,:,z3])

                    gridSlice = np.sum([gridSlice1, gridSlice2, gridSlice3], axis=0)
                    grid = gridSlice


                    empty = np.zeros((mapObj.xDivisions, mapObj.yDivisions),dtype=np.float32)

                    img = cv2.merge((grid, empty, empty))
                    img = cv2.transpose(img)

                    img = cv2.circle(img, (x, y), 5, (0, 1, 0), 2)

                    vec = np.asarray([20, 0, 0])
                    vec = r.apply(vec)  # Aero-ref -> Aero-body


                    vec[0] += x 
                    vec[1] += y

                    img = cv2.line(img, (x, y), (int(vec[0]), int(vec[1])), (0, 0, 1), 2)
                    img = cv2.resize(img, (540, 540))
                    cv2.imshow('map', img)
                    cv2.waitKey(1)
                    t16 = time.perf_counter()
                    print(f"get frames, deproject and update map and visualise: {t16 - t13:0.4f} seconds")
                    # time.sleep(0.5)

                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    traceback.print_exc(file=sys.stdout)
                    
        except KeyboardInterrupt:
            pass
