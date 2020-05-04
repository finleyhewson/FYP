def main():
    print("Start Hybrid A* planning")
    import time
    import Point_Cloud as map
    import T265_Tracking_Camera as t265
    import D435_Depth_Camera as d435
    import Hybrid_Astar as planner
    import vision_to_mavros
    import cv2
    import base64
    import threading
    import copy
    import traceback
    import numpy as np
    import matplotlib.pyplot as plt
    import heapq
    import scipy.spatial
    import sys
    import math
if __name__ == "__main__":
    print("*** STARTING ***")

    parser = vision_to_mavros.GetParser()
    args = parser.parse_args()

    connection_string = args.connect
    connection_baudrate = args.baudrate
    vision_position_estimate_msg_hz = args.vision_position_estimate_msg_hz
    vision_position_delta_msg_hz = args.vision_position_delta_msg_hz
    scale_calib_enable = args.scale_calib_enable
    camera_orientation = args.camera_orientation
    debug_enable = args.debug_enable
    = args.Point_Cloud

    t265Obj = t265.rs_t265()
    d435Obj = d435.rs_d435(framerate=30, width=480, height=270)
    mapObj = map.mapper()
    s = 0 # used to create initial path planning loop
    XY_GRID_RESOLUTION = 0.25  # [m]
    VR = 0.2  # robot radius
    with t265Obj, d435Obj:
        try:
            while True: # while pos isn't within a certain distance of the goal position try this, 
                #once it is input a new goal point, minus the previous goal point from current one to keep within radius

                tik=time.perf_counter()
                # Get frames of data - points and global 6dof
                pos, r, conf, _ = t265Obj.get_frame()
                frame, rgbImg = d435Obj.getFrame()
                points = d435Obj.deproject_frame(frame)
                mapObj.update(points, pos, r)

                try:
                 
                    x = np.digitize(pos[0], mapObj.xBins) - 1
                    y = np.digitize(pos[1], mapObj.yBins) - 1
                    z = np.digitize(pos[2], mapObj.zBins) - 1
                    z2= np.digitize(pos[2], mapObj.zBins) - 2
                    z3= np.digitize(pos[2], mapObj.zBins) - 0

                    #Taking a slice of the obstacles x and y coordinates at three different z heights
                    gridSlice1=copy.copy(mapObj.grid[:,:,z])
                    gridSlice2=copy.copy(mapObj.grid[:,:,z2])
                    gridSlice3=copy.copy(mapObj.grid[:,:,z3])

                    #Adding the slices together
                    gridSlice = np.sum([gridSlice1, gridSlice2, gridSlice3], axis=0)
                    grid = gridSlice
                    
                    #Creating an image that can be displayed of the obstacles  
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

                    #Defining x and y coordinates of obstacles
                    ox, oy = [], []

                    #Using this to define the maximum search area of the path planner
                    for i in np.arange(-5,5,0.5):
                        ox.append(i)
                        oy.append(-5)
                    for i in np.arange(-5,5,0.5):
                        ox.append(5)
                        oy.append(i)
                    for i in np.arange(-5,5.5,0.5):
                        ox.append(i)
                        oy.append(5)
                    for i in np.arange(-5,5,0.5):
                        ox.append(-5)
                        oy.append(i)
                    
                    #Appending obstacles x and y coordinates from grid
                    grid = cv2.transpose(grid)
                    for i in range(grid.shape[0]):
                        for j in range(grid.shape[1]):
                            if grid[i][j] > 0:
                                ox.append(mapObj.xBins[i])
                                oy.append(mapObj.yBins[j])

                    # Should have North as 90 degrees
                    # Set Initial parameters
                    # Start position is always the current position of the car
                    # Need to have a way of making the function still generate a path if the start is within range of an obstacle
                    yaw_angle = r.as_euler('zyx', degrees=True)
                    start = [pos[0], pos[1], np.deg2rad(90.0 - yaw_angle[0])]
                    goal = [0.0, 3.0, np.deg2rad(90.0)] #90 faces to the top, 0 to the right, -90 towards the bottom
    

                    plt.plot(ox, oy, ".k")
                    planner.rs.plot_arrow(start[0], start[1], start[2], fc='g')
                    planner.rs.plot_arrow(goal[0], goal[1], goal[2])
                    plt.grid(True)
                    plt.axis("equal")

                    #initial path calculation loop
                    if s == 0:
                        
                        #path planner function
                        path = planner.hybrid_a_star_planning(start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
                        tok=time.perf_counter()
                        print(f"Path Planner in {tok - tik:0.4f} seconds")

                        #list of x,y,yaw and direction for the path
                        xpath = path.xlist
                        ypath = path.ylist
                        yawpath = path.yawlist
                        directionpath = path.directionlist
                        s=s+1
                        #display path
                        for ix, iy, iyaw in zip(xpath, ypath, yawpath):
                            plt.cla()
                            plt.plot(ox, oy, ".k")
                            plt.plot(xpath, ypath, "-r", label="Hybrid A* path")
                            plt.grid(True)
                            plt.axis("equal")
                            plot_car(ix, iy, iyaw)
                            plt.pause(0.0001)

                        print(__file__ + " done!!")
                    
                    #second path planner loop, new path only calculated if obstacle detected in route of old path
                    elif s != 0:

                        #use the obstacle map to check if any of the new obstacles will cause a collision with the path
                        ox1 = [iox / XY_GRID_RESOLUTION for iox in ox] 
                        oy1 = [ioy / XY_GRID_RESOLUTION for ioy in oy]
                        obmap, minx, miny, maxx, maxy, xw, yw = planner.calc_obstacle_map(ox1, oy1, XY_GRID_RESOLUTION, VR)

                        #runs through the initial path x and y values and creates a new path if they're within range of an obstacle 
                        for ind in range(len(path.xlist)): 
                            if obmap[int(round((path.xlist[ind]/XY_GRID_RESOLUTION) - minx))][int(round((path.ylist[ind]/XY_GRID_RESOLUTION) - miny))]:
                                #generate a new path
                                path = planner.hybrid_a_star_planning(start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
                                tok=time.perf_counter()
                                print(f"Path Planner in {tok - tik:0.4f} seconds")

                                #list of x,y,yaw and direction of the new path
                                xpath = path.xlist
                                ypath = path.ylist
                                yawpath = path.yawlist
                                directionpath = path.directionlist

                                for ix, iy, iyaw in zip(xpath, ypath, yawpath):
                                    plt.cla()
                                    plt.plot(ox, oy, ".k")
                                    plt.plot(xpath, ypath, "-r", label="Hybrid A* path")
                                    plt.grid(True)
                                    plt.axis("equal")
                                    plot_car(ix, iy, iyaw)
                                    plt.pause(0.0001)
                                print(__file__ + " done!!")
                                break
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    traceback.print_exc(file=sys.stdout)
        except KeyboardInterrupt:
            pass

