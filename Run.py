import Point_Cloud as map
import T265_Tracking_Camera as t265
import D435_Depth_Camera as d435
import Hybrid_Astar as planner
import Motion
import Pixhawk
import Position
import Arg_Parser
import time
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
sys.path.append("/usr/local/lib/")

# Set MAVLink protocol to 2.
import os
os.environ["MAVLINK20"] = "1"

# Import the libraries
import pyrealsense2 as rs
import threading
from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler
from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil
if __name__ == "__main__":
    print("*** STARTING ***")

    pixhawkObj = Pixhawk.Pix()

    parser = Arg_Parser.GetParser()
    args = parser.parse_args()

    connection_string = args.connect
    connection_baudrate = args.baudrate
    vision_position_estimate_msg_hz = args.vision_position_estimate_msg_hz
    scale_calib_enable = args.scale_calib_enable
    camera_orientation = args.camera_orientation
    debug_enable = args.debug_enable

    # Using default values if no specified inputs
    if not connection_string:
        connection_string = pixhawkObj.connection_string_default
        print("INFO: Using default connection_string", connection_string)
    else:
        print("INFO: Using connection_string", connection_string)

    if not connection_baudrate:
        connection_baudrate = pixhawkObj.connection_baudrate_default
        print("INFO: Using default connection_baudrate", connection_baudrate)
    else:
        print("INFO: Using connection_baudrate", connection_baudrate)

    if not vision_position_estimate_msg_hz:
        vision_position_estimate_msg_hz = pixhawkObj.vision_position_estimate_msg_hz_default
        print("INFO: Using default vision_position_estimate_msg_hz", vision_position_estimate_msg_hz)
    else:
        print("INFO: Using vision_position_estimate_msg_hz", vision_position_estimate_msg_hz)

    if pixhawkObj.body_offset_enabled == 1:
        print("INFO: Using camera position offset: Enabled, x y z is", pixhawkObj.body_offset_x, pixhawkObj.body_offset_y, pixhawkObj.body_offset_z)
    else:
        print("INFO: Using camera position offset: Disabled")

    if pixhawkObj.compass_enabled == 1:
        print("INFO: Using compass: Enabled. Heading will be aligned to north.")
    else:
        print("INFO: Using compass: Disabled")

    if scale_calib_enable == True:
        print("\nINFO: SCALE CALIBRATION PROCESS. DO NOT RUN DURING FLIGHT.\nINFO: TYPE IN NEW SCALE IN FLOATING POINT FORMAT\n")
    else:
        if pixhawkObj.scale_factor == 1.0:
            print("INFO: Using default scale factor", pixhawkObj.scale_factor)
        else:
            print("INFO: Using scale factor", pixhawkObj.scale_factor)

    if not camera_orientation:
        camera_orientation = pixhawkObj.camera_orientation_default
        print("INFO: Using default camera orientation", camera_orientation)
    else:
        print("INFO: Using camera orientation", camera_orientation)
    if camera_orientation == 0:     # Forward, USB port to the right
        H_aeroRef_T265Ref   = np.array([[0,0,-1,0],[1,0,0,0],[0,-1,0,0],[0,0,0,1]])
        H_T265body_aeroBody = np.linalg.inv(H_aeroRef_T265Ref)
    else:                           # Default is facing forward, USB port to the right
        H_aeroRef_T265Ref   = np.array([[0,0,-1,0],[1,0,0,0],[0,-1,0,0],[0,0,0,1]])
        H_T265body_aeroBody = np.linalg.inv(H_aeroRef_T265Ref)

    if not debug_enable:
        debug_enable = 0
    else:
        debug_enable = 1
        np.set_printoptions(precision=4, suppress=True) # Format output on terminal 
        print("INFO: Debug messages enabled.")

        
    print("INFO: Connecting to vehicle.")
    while (not pixhawkObj.vehicle_connect(connection_string, connection_baudrate)):
        pass
    print("INFO: Vehicle connected.")
    

    d435Obj = d435.rs_d435(framerate=30, width=480, height=270)
    posObj = Position.position(pixhawkObj)
    mapObj = map.mapper()
    motionObj = Motion.motion(pixhawkObj)

    #Schedules Mavlink Messages in the Background at predetermined frequencies
    sched = BackgroundScheduler()

    if pixhawkObj.enable_msg_vision_position_estimate or pixhawkObj.enable_update_tracking_confidence_to_gcs:
        sched.add_job(posObj.loop, 'interval', seconds = 1/vision_position_estimate_msg_hz)


    # A separate thread to monitor user input
    user_keyboard_input_thread = threading.Thread(target=pixhawkObj.user_input_monitor)
    user_keyboard_input_thread.daemon = True
    user_keyboard_input_thread.start()

    sched.start()
    print("INFO: Press Enter to set EKF home at default location")

    pixhawkObj.set_default_global_origin()
    pixhawkObj.set_default_home_position()

    # x, y, yaw, of the waypoints taken from global path planner
    waypoints = np.asarray([[0.0, 4.0, 90.0],
            [0.0, 0.0, 180.0]])

    s = 0 # used to create initial path planning loop
    with d435Obj:
        try:
          while True:
            # Monitor last_heartbeat to reconnect in case of lost connection
            if pixhawkObj.vehicle.last_heartbeat > pixhawkObj.connection_timeout_sec_default:
                pixhawkObj.is_vehicle_connected = False
                print("WARNING: CONNECTION LOST. Last hearbeat was %f sec ago."% pixhawkObj.vehicle.last_heartbeat)
                print("WARNING: Attempting to reconnect ...")
                pixhawkObj.vehicle_connect()
                continue

            for iway in range(len(waypoints)):

                curr_goal = waypoints[iway]
                curr_pos, _, _ = posObj.update()

                remaining_distance = math.sqrt((curr_goal[0] - curr_pos[0])**2 + (curr_goal[1] - curr_pos[1])**2)
                close_enough = 0.1

                while remaining_distance > close_enough: # some amount of distance away from the curr_position

                    # Get frames of data - points and global 6dof
                    pos, r, conf = posObj.update()
                    remaining_distance = math.sqrt((curr_goal[0] - pos[0])**2 + (curr_goal[1]- pos[1])**2)

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
                            if np.max(grid[i]) > 0.0:
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
                        goal = [curr_goal[0], curr_goal[1], np.deg2rad(90.0 - curr_goal[2])] #90 faces to the top, 0 to the right, -90 towards the bottom

                        #if path_index == 0:
                        #    path_index = path_index + 1
                        #else:
                        #    if path_index - 1 >= len(xpath):
                        #        break
                        #    else:
                        #        xvel = xpath[path_index - 1]
                        #        yvel = ypath[path_index - 1]
                        #        yaw_set = yawpath[path_index -1]

                        #        #needs to be faster 0.1m every sec is slow as shit
                        #        pixhawkObj.send_ned_position(xvel, yvel, 0 , 1)
                        #        pixhawkObj.condition_yaw(yaw_set, relative = False)

                        #        path_index = path_index + 1

                        #initial path calculation loop
                        if s == 0:
                        
                            #path planner function
                            path = planner.hybrid_a_star_planning(start, goal, ox, oy, planner.XY_GRID_RESOLUTION, planner.YAW_GRID_RESOLUTION)
                            

                            #list of x,y,yaw and direction for the path
                            xpath = path.xlist
                            ypath = path.ylist
                            yawpath = path.yawlist
                            directionpath = path.directionlist

                            #creating thread for motion
                            motionObj.update(xpath, ypath, yawpath)
                            motion_thread = threading.Thread(target=motionObj.loop)
                            motion_thread.daemon = True
                            motion_thread.start()

                            s=s+1
                            
                    
                        #second path planner loop, new path only calculated if obstacle detected in route of old path
                        elif s != 0:

                            #use the obstacle map to check if any of the new obstacles will cause a collision with the path
                            ox1 = [iox / planner.XY_GRID_RESOLUTION for iox in ox] 
                            oy1 = [ioy / planner.XY_GRID_RESOLUTION for ioy in oy]
                            obmap, minx, miny, maxx, maxy, xw, yw = planner.calc_obstacle_map(ox1, oy1, planner.XY_GRID_RESOLUTION, planner.VR)

                            #runs through the initial path x and y values and creates a new path if they're within range of an obstacle 
                            for ind in range(len(path.xlist)): 
                                if obmap[int(round((path.xlist[ind]/planner.XY_GRID_RESOLUTION) - minx))][int(round((path.ylist[ind]/planner.XY_GRID_RESOLUTION) - miny))]:
                                    
                                    # send command to the pixhawk to stop moving and wait for new path to be calculated
                                    motionObj.recalc(recalc_path = True)

                                    # plan a new path
                                    path = planner.hybrid_a_star_planning(start, goal, ox, oy, planner.XY_GRID_RESOLUTION, planner.YAW_GRID_RESOLUTION)
                                    
                                    #list of x,y,yaw and direction of the new path
                                    xpath = path.xlist
                                    ypath = path.ylist
                                    yawpath = path.yawlist
                                    directionpath = path.directionlist

                                    # update pixhawk of the directions of the new path
                                    motionObj.update(xpath, ypath, yawpath)
                                    MotionObj.recalc(recalc_path = False)

                                    break
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except:
                        traceback.print_exc(file=sys.stdout)
        except KeyboardInterrupt:
            pass
        
        finally:
            motionObj.close()
            d435Obj.closeConnection()
            pixhawkObj.vehicle.close()
            print("INFO: Realsense pipeline and vehicle object closed.")
            sys.exit()

