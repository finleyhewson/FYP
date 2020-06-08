#!/usr/bin/env python3

#####################################################
##          librealsense T265 to MAVLink           ##
#####################################################
# This script assumes pyrealsense2.[].so file is found under the same directory as this script
# Install required packages: 
#   pip install pyrealsense2
#   pip install transformations
#   pip3 install dronekit
#   pip3 install apscheduler

# Set the path for IDLE
import sys
import dronekit
sys.path.append("/usr/local/lib/")

# Set MAVLink protocol to 2.
import os
os.environ["MAVLINK20"] = "1"

# Import the libraries
import pyrealsense2 as rs
import numpy as np
#import transformations as tf
import math as m
import time
import scipy.spatial
import argparse
import threading
from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler
from dronekit import connect, VehicleMode
from pymavlink import mavutil

#######################################
# Parameters
#######################################
class Pix:
    def __init__(self):
        ########Anything with self, you're going to have to change
        # Default configurations for connection to the FCU
        self.connection_string_default = '/dev/ttyUSB0'
        self.connection_baudrate_default = 921600
        self.connection_timeout_sec_default = 5

        self.camera_orientation_default = 0

        # Enable/disable each message/function individually
        self.enable_msg_vision_position_estimate = True
        self.vision_position_estimate_msg_hz_default = 15

        #self.enable_msg_vision_position_delta = False
        #self.vision_position_delta_msg_hz_default = 15

        self.enable_update_tracking_confidence_to_gcs = True
        self.update_tracking_confidence_to_gcs_hz_default = 1

        # Default global position for EKF home/ origin
        self.enable_auto_set_ekf_home = False
        self.home_lat = 151269321    # Somewhere random
        self.home_lon = 16624301     # Somewhere random
        self.home_alt = 163000       # Somewhere random

        # TODO: Taken care of by ArduPilot, so can be removed (once the handling on AP side is confirmed stable)
        # In NED frame, offset from the IMU or the center of gravity to the camera's origin point
        self.body_offset_enabled = 0
        self.body_offset_x = 0  # In meters (m)
        self.body_offset_y = 0  # In meters (m)
        self.body_offset_z = 0  # In meters (m)

        # Global scale factor, position x y z will be scaled up/down by this factor
        self.scale_factor = 1.0

        # Enable using yaw from compass to align north (zero degree is facing north)
        self.compass_enabled = 0

        # pose data confidence: 0x0 - Failed / 0x1 - Low / 0x2 - Medium / 0x3 - High 
        self.pose_data_confidence_level = ('Failed', 'Low', 'Medium', 'High')

        # lock for thread synchronization
        self.lock = threading.Lock()

        #######################################
        # Global variables
        #######################################

        # FCU connection variables
        self.vehicle = None
        self.is_vehicle_connected = False

        # Camera-related variables
        self.pipe = None

        # Data variables
        self.data = None
        self.H_aeroRef_aeroBody = None
        self.heading_north_yaw = None
        self.current_confidence_level = None
        

        #######################################
        # Parsing user' inputs
        #######################################
        #def GetParser():
        #    parser = argparse.ArgumentParser(description='Reboots vehicle')
        #    parser.add_argument('--connect',
        #                        help="Vehicle connection target string. If not specified, a default string will be used.")
        #    parser.add_argument('--baudrate', type=float,
        #                        help="Vehicle connection baudrate. If not specified, a default value will be used.")
        #    parser.add_argument('--vision_position_estimate_msg_hz', type=float,
        #                        help="Update frequency for VISION_POSITION_ESTIMATE message. If not specified, a default value will be used.")
        #    parser.add_argument('--vision_position_delta_msg_hz', type=float,
        #                        help="Update frequency for VISION_POSITION_DELTA message. If not specified, a default value will be used.")
        #    parser.add_argument('--scale_calib_enable', default=False, action='store_true',
        #                        help="Scale calibration. Only run while NOT in flight")
        #    parser.add_argument('--camera_orientation', type=int,
        #                        help="Configuration for camera orientation. Currently supported: forward, usb port to the right - 0; downward, usb port to the right - 1, 2: forward tilted down 45deg")
        #    parser.add_argument('--debug_enable',type=int,
        #                        help="Enable debug messages on terminal")
        #    parser.add_argument( '--path_plan', '-p', help = 'Enable collision avoidance', default = None, action = "store_true", required=False)

        #    return parser
        #parser = GetParser()
        #args = parser.parse_args()

        #connection_string = args.connect
        #connection_baudrate = args.baudrate
        #vision_position_estimate_msg_hz = args.vision_position_estimate_msg_hz
        #vision_position_delta_msg_hz = args.vision_position_delta_msg_hz
        #scale_calib_enable = args.scale_calib_enable
        #camera_orientation = args.camera_orientation
        #debug_enable = args.debug_enable
        #path_planner = args.path_plan

        # Using default values if no specified inputs
        #if not connection_string:
        #    connection_string = connection_string_default
        #    print("INFO: Using default connection_string", connection_string)
        #else:
        #    print("INFO: Using connection_string", connection_string)

        #if not connection_baudrate:
        #    connection_baudrate = connection_baudrate_default
        #    print("INFO: Using default connection_baudrate", connection_baudrate)
        #else:
        #    print("INFO: Using connection_baudrate", connection_baudrate)

        #if not vision_position_estimate_msg_hz:
        #    vision_position_estimate_msg_hz = vision_position_estimate_msg_hz_default
        #    print("INFO: Using default vision_position_estimate_msg_hz", vision_position_estimate_msg_hz)
        #else:
        #    print("INFO: Using vision_position_estimate_msg_hz", vision_position_estimate_msg_hz)
    
        #if not vision_position_delta_msg_hz:
        #    vision_position_delta_msg_hz = vision_position_delta_msg_hz_default
        #    print("INFO: Using default vision_position_delta_msg_hz", vision_position_delta_msg_hz)
        #else:
        #    print("INFO: Using vision_position_delta_msg_hz", vision_position_delta_msg_hz)

        #if body_offset_enabled == 1:
        #    print("INFO: Using camera position offset: Enabled, x y z is", body_offset_x, body_offset_y, body_offset_z)
        #else:
        #    print("INFO: Using camera position offset: Disabled")

        #if compass_enabled == 1:
        #    print("INFO: Using compass: Enabled. Heading will be aligned to north.")
        #else:
        #    print("INFO: Using compass: Disabled")

        #if scale_calib_enable == True:
        #    print("\nINFO: SCALE CALIBRATION PROCESS. DO NOT RUN DURING FLIGHT.\nINFO: TYPE IN NEW SCALE IN FLOATING POINT FORMAT\n")
        #else:
        #    if scale_factor == 1.0:
        #        print("INFO: Using default scale factor", scale_factor)
        #    else:
        #        print("INFO: Using scale factor", scale_factor)

        #if not camera_orientation:
        #    camera_orientation = camera_orientation_default
        #    print("INFO: Using default camera orientation", camera_orientation)
        #else:
        #    print("INFO: Using camera orientation", camera_orientation)

        # Transformation to convert different camera orientations to NED convention. Replace camera_orientation_default for your configuration.
        #   0: Forward, USB port to the right
        #   1: Downfacing, USB port to the right 
        #   2: Forward, 45 degree tilted down
        # Important note for downfacing camera: you need to tilt the vehicle's nose up a little - not flat - before you run the script, otherwise the initial yaw will be randomized, read here for more details: https://github.com/IntelRealSense/librealsense/issues/4080. Tilt the vehicle to any other sides and the yaw might not be as stable.

        #if camera_orientation == 0:     # Forward, USB port to the right
        #    H_aeroRef_T265Ref   = np.array([[0,0,-1,0],[1,0,0,0],[0,-1,0,0],[0,0,0,1]])
        #    H_T265body_aeroBody = np.linalg.inv(H_aeroRef_T265Ref)
        #elif camera_orientation == 1:   # Downfacing, USB port to the right
        #    H_aeroRef_T265Ref   = np.array([[0,0,-1,0],[1,0,0,0],[0,-1,0,0],[0,0,0,1]])
        #    H_T265body_aeroBody = np.array([[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]])
        #elif camera_orientation == 2:   # 45degree forward
        #    H_aeroRef_T265Ref   = np.array([[0,0,-1,0],[1,0,0,0],[0,-1,0,0],[0,0,0,1]])
        #    H_T265body_aeroBody = (tf.euler_matrix(m.pi/4, 0, 0)).dot(np.linalg.inv(H_aeroRef_T265Ref))
        #else:                           # Default is facing forward, USB port to the right
        #    H_aeroRef_T265Ref   = np.array([[0,0,-1,0],[1,0,0,0],[0,-1,0,0],[0,0,0,1]])
        #    H_T265body_aeroBody = np.linalg.inv(H_aeroRef_T265Ref)

        #if not debug_enable:
        #    debug_enable = 0
        #else:
        #    debug_enable = 1
        #    np.set_printoptions(precision=4, suppress=True) # Format output on terminal 
        #    print("INFO: Debug messages enabled.")


    #######################################
    # Functions
    #######################################

    def send_ned_position(self, xpath, ypath, zpath):
        """
        Move vehicle in direction based on specified velocity vectors.
        """
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,       # time_boot_ms (not used)
            0, 0,    # target system, target component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, # frame
            0b0000111111111000, # type_mask (only speeds enabled)
            xpath, ypath, zpath, # x, y, z positions 
            0, 0, 0, # x, y, z velocity in m/s (not used)
            0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
            0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
        
        # send command
        self.vehicle.send_mavlink(msg)

    def setmode(self):
        self.vehicle.mode = VehicleMode('GUIDED')
        print("Vehicle mode set to Guided")



    def condition_yaw(self, heading, relative=False):
        """
        Send MAV_CMD_CONDITION_YAW message to point vehicle at a specified heading (in degrees).

        This method sets an absolute heading by default, but you can set the `relative` parameter
        to `True` to set yaw relative to the current yaw heading.

        By default the yaw of the vehicle will follow the direction of travel. After setting 
        the yaw using this function there is no way to return to the default yaw "follow direction 
        of travel" behaviour (https://github.com/diydrones/ardupilot/issues/2427)

        For more information see: 
        http://copter.ardupilot.com/wiki/common-mavlink-mission-command-messages-mav_cmd/#mav_cmd_condition_yaw
        """
        if relative:
            is_relative = 1 #yaw relative to direction of travel
        else:
            is_relative = 0 #yaw is an absolute angle
        # create the CONDITION_YAW command using command_long_encode()
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,    # target system, target component
            mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
            0, #confirmation
            heading,    # param 1, yaw in degrees
            0,          # param 2, yaw speed deg/s
            1,          # param 3, direction -1 ccw, 1 cw
            is_relative, # param 4, relative offset 1, absolute angle 0
            0, 0, 0)    # param 5 ~ 7 not used
        # send command to vehicle
        self.vehicle.send_mavlink(msg)


    # https://mavlink.io/en/messages/common.html#VISION_POSITION_ESTIMATE
    def send_vision_position_estimate_message(self, _pos, _r, current_time_us):
        global is_vehicle_connected
        with self.lock:
            if self.is_vehicle_connected == True:
                rot_eul_angles = _r.as_euler('zyx', degrees=False)
                msg = self.vehicle.message_factory.vision_position_estimate_encode(
                    current_time_us,                    # us Timestamp (UNIX time or time since system boot)
                    _pos[0],	        # Global X position
                    _pos[1],           # Global Y position
                    _pos[2],	        # Global Z position
                    rot_eul_angles[2],	                        # Roll angle
                    rot_eul_angles[1],	                        # Pitch angle
                    rot_eul_angles[0]	                        # Yaw angle
                )
                self.vehicle.send_mavlink(msg)
                self.vehicle.flush()

    # https://mavlink.io/en/messages/ardupilotmega.html#VISION_POSITION_DELTA
    #def send_vision_position_delta_message(self, H_aeroRef_aeroBody):
    #    global is_vehicle_connected, current_time_us, current_confidence_level
    #    with self.lock:
    #        if self.is_vehicle_connected == True and H_aeroRef_aeroBody is not None:
    #            # Calculate the deltas in position, attitude and time from the previous to current orientation
    #            H_aeroRef_PrevAeroBody      = send_vision_position_delta_message.H_aeroRef_PrevAeroBody
    #            H_PrevAeroBody_CurrAeroBody = (np.linalg.inv(H_aeroRef_PrevAeroBody)).dot(H_aeroRef_aeroBody)

    #            delta_time_us    = current_time_us - send_vision_position_delta_message.prev_time_us
    #            delta_position_m = [H_PrevAeroBody_CurrAeroBody[0][3], H_PrevAeroBody_CurrAeroBody[1][3], H_PrevAeroBody_CurrAeroBody[2][3]]
    #            delta_angle_rad  = np.array( tf.euler_from_matrix(H_PrevAeroBody_CurrAeroBody, 'sxyz'))

    #            # Send the message
    #            msg = vehicle.message_factory.vision_position_delta_encode(
    #                current_time_us,    # us: Timestamp (UNIX time or time since system boot)
    #                delta_time_us,	    # us: Time since last reported camera frame
    #                delta_angle_rad,    # float[3] in radian: Defines a rotation vector in body frame that rotates the vehicle from the previous to the current orientation
    #                delta_position_m,   # float[3] in m: Change in position from previous to current frame rotated into body frame (0=forward, 1=right, 2=down)
    #                current_confidence_level # Normalised confidence value from 0 to 100. 
    #            )
    #            vehicle.send_mavlink(msg)
    #            vehicle.flush()

    #            # Save static variables
    #            send_vision_position_delta_message.H_aeroRef_PrevAeroBody = H_aeroRef_aeroBody
    #            send_vision_position_delta_message.prev_time_us = current_time_us

    def send_tracking_confidence_to_gcs(self, _conf):
        global current_confidence_level
        confidence_status_string = 'Tracking confidence: ' + self.pose_data_confidence_level[_conf]
        self.send_msg_to_gcs(confidence_status_string)

    def send_msg_to_gcs(self, text_to_be_sent):
        # MAV_SEVERITY: 0=EMERGENCY 1=ALERT 2=CRITICAL 3=ERROR, 4=WARNING, 5=NOTICE, 6=INFO, 7=DEBUG, 8=ENUM_END
        # Defined here: https://mavlink.io/en/messages/common.html#MAV_SEVERITY
        # MAV_SEVERITY = 3 will let the message be displayed on Mission Planner HUD, but 6 is ok for QGroundControl
        if self.is_vehicle_connected == True:
            text_msg = 'T265: ' + text_to_be_sent
            status_msg = self.vehicle.message_factory.statustext_encode(
                6,                      # MAV_SEVERITY
                text_msg.encode()	    # max size is char[50]       
            )
            self.vehicle.send_mavlink(status_msg)
            self.vehicle.flush()
            print("INFO: " + text_to_be_sent)
        else:
            print("INFO: Vehicle not connected. Cannot send text message to Ground Control Station (GCS)")

    # Send a mavlink SET_GPS_GLOBAL_ORIGIN message (http://mavlink.org/messages/common#SET_GPS_GLOBAL_ORIGIN), which allows us to use local position information without a GPS.
    def set_default_global_origin(self):
        if self.is_vehicle_connected == True:
            msg = self.vehicle.message_factory.set_gps_global_origin_encode(
                int(self.vehicle._master.source_system),
                self.home_lat, 
                self.home_lon,
                self.home_alt
            )

            self.vehicle.send_mavlink(msg)
            self.vehicle.flush()

    # Send a mavlink SET_HOME_POSITION message (http://mavlink.org/messages/common#SET_HOME_POSITION), which allows us to use local position information without a GPS.
    def set_default_home_position(self):
        if self.is_vehicle_connected == True:
            x = 0
            y = 0
            z = 0
            q = [1, 0, 0, 0]   # w x y z

            approach_x = 0
            approach_y = 0
            approach_z = 1

            msg = self.vehicle.message_factory.set_home_position_encode(
                int(self.vehicle._master.source_system),
                self.home_lat, 
                self.home_lon,
                self.home_alt,
                x,
                y,
                z,
                q,
                approach_x,
                approach_y,
                approach_z
            )

            self.vehicle.send_mavlink(msg)
            self.vehicle.flush()

    # Request a timesync update from the flight controller, for future work.
    # TODO: Inspect the usage of timesync_update 
    def update_timesync(self, ts=0, tc=0):
        if ts == 0:
            ts = int(round(time.time() * 1000))
        msg = vehicle.message_factory.timesync_encode(
            tc,     # tc1
            ts      # ts1
        )
        vehicle.send_mavlink(msg)
        vehicle.flush()

    # Listen to attitude data to acquire heading when compass data is enabled
    def att_msg_callback(self, attr_name, value):
        global heading_north_yaw
        if heading_north_yaw is None:
            heading_north_yaw = value.yaw
            print("INFO: Received first ATTITUDE message with heading yaw", heading_north_yaw * 180 / m.pi, "degrees")
        else:
            heading_north_yaw = value.yaw
            print("INFO: Received ATTITUDE message with heading yaw", heading_north_yaw * 180 / m.pi, "degrees")


    def vehicle_connect(self, connection_string, connection_baudrate):
        global vehicle, is_vehicle_connected
    
        try:
            self.vehicle = connect(connection_string, wait_ready = True, baud = connection_baudrate, source_system = 1)
        except:
            print('Connection error! Retrying...')
            sleep(1)

        if self.vehicle == None:
            self.is_vehicle_connected = False
            return False
        else:
            self.is_vehicle_connected = True
            return True


    #def realsense_connect(self):
    #    global pipe
    #    # Declare RealSense pipeline, encapsulating the actual device and sensors
    #    pipe = rs.pipeline()

    #    # Build config object before requesting data
    #    cfg = rs.config()

    #    # Enable the stream we are interested in
    #    cfg.enable_stream(rs.stream.pose) # Positional data 

    #    # Start streaming with requested config
    #    pipe.start(cfg)


    # Monitor user input from the terminal and perform action accordingly
    def user_input_monitor(self): #, scale_calib_enable):
        global scale_factor
        while True:
            # Specical case: updating scale
            #if scale_calib_enable == True:
            #    scale_factor = float(input("INFO: Type in new scale as float number\n"))
            #    print("INFO: New scale is ", scale_factor)

            if self.enable_auto_set_ekf_home:
                self.send_msg_to_gcs('Set EKF home with default GPS location')
                self.set_default_global_origin()
                self.set_default_home_position()
                time.sleep(1) # Wait a short while for FCU to start working

            # Add new action here according to the key pressed.
            # Enter: Set EKF home when user press enter
            try:
                c = input()
                if c == "":
                    self.send_msg_to_gcs('Set EKF home with default GPS location')
                    self.set_default_global_origin()
                    self.set_default_home_position()
                else:
                    print("Got keyboard input", c)
            except IOError: pass


#######################################
# Main code starts here
#######################################

#print("INFO: Connecting to vehicle.")
#while (not vehicle_connect()):
#    pass
#print("INFO: Vehicle connected.")

#send_msg_to_gcs('Connecting to camera...')
#realsense_connect()
#send_msg_to_gcs('Camera connected.')

#if compass_enabled == 1:
#    # Listen to the attitude data in aeronautical frame
#    vehicle.add_message_listener('ATTITUDE', att_msg_callback)

## Send MAVlink messages in the background at pre-determined frequencies
#sched = BackgroundScheduler()

#if enable_msg_vision_position_estimate:
#    sched.add_job(send_vision_position_estimate_message, 'interval', seconds = 1/vision_position_estimate_msg_hz)

#if enable_msg_vision_position_delta:
#    sched.add_job(send_vision_position_delta_message, 'interval', seconds = 1/vision_position_delta_msg_hz)
#    send_vision_position_delta_message.H_aeroRef_PrevAeroBody = tf.quaternion_matrix([1,0,0,0]) 
#    send_vision_position_delta_message.prev_time_us = int(round(time.time() * 1000000))

#if enable_update_tracking_confidence_to_gcs:
#    sched.add_job(send_tracking_confidence_to_gcs, 'interval', seconds = 1/update_tracking_confidence_to_gcs_hz_default)

## A separate thread to monitor user input
#user_keyboard_input_thread = threading.Thread(target=user_input_monitor)
#user_keyboard_input_thread.daemon = True
#user_keyboard_input_thread.start()

#sched.start()

#if compass_enabled == 1:
#    time.sleep(1) # Wait a short while for yaw to be correctly initiated

#send_msg_to_gcs('Sending vision messages to FCU')

#print("INFO: Press Enter to set EKF home at default location")

#try:
#    while True:
#        # Monitor last_heartbeat to reconnect in case of lost connection
#        if vehicle.last_heartbeat > connection_timeout_sec_default:
#            is_vehicle_connected = False
#            print("WARNING: CONNECTION LOST. Last hearbeat was %f sec ago."% vehicle.last_heartbeat)
#            print("WARNING: Attempting to reconnect ...")
#            vehicle_connect()
#            continue
        
#        # Wait for the next set of frames from the camera
#        frames = pipe.wait_for_frames()

#        # Fetch pose frame
#        pose = frames.get_pose_frame()

#        # Process data
#        if pose:
#            with self.lock:
#                # Store the timestamp for MAVLink messages
#                current_time_us = int(round(time.time() * 1000000))

#                # Pose data consists of translation and rotation
#                data = pose.get_pose_data()
                
#                # Confidence level value from T265: 0-3, remapped to 0 - 100: 0% - Failed / 33.3% - Low / 66.6% - Medium / 100% - High  
#                current_confidence_level = float(data.tracker_confidence * 100 / 3)  

#                # In transformations, Quaternions w+ix+jy+kz are represented as [w, x, y, z]!
#                H_T265Ref_T265body = tf.quaternion_matrix([data.rotation.w, data.rotation.x, data.rotation.y, data.rotation.z]) 
#                H_T265Ref_T265body[0][3] = data.translation.x * scale_factor
#                H_T265Ref_T265body[1][3] = data.translation.y * scale_factor
#                H_T265Ref_T265body[2][3] = data.translation.z * scale_factor

#                # Transform to aeronautic coordinates (body AND reference frame!)
#                H_aeroRef_aeroBody = H_aeroRef_T265Ref.dot( H_T265Ref_T265body.dot( H_T265body_aeroBody))

#                # Take offsets from body's center of gravity (or IMU) to camera's origin into account
#                if body_offset_enabled == 1:
#                    H_body_camera = tf.euler_matrix(0, 0, 0, 'sxyz')
#                    H_body_camera[0][3] = body_offset_x
#                    H_body_camera[1][3] = body_offset_y
#                    H_body_camera[2][3] = body_offset_z
#                    H_camera_body = np.linalg.inv(H_body_camera)
#                    H_aeroRef_aeroBody = H_body_camera.dot(H_aeroRef_aeroBody.dot(H_camera_body))

#                # Realign heading to face north using initial compass data
#                if compass_enabled == 1:
#                    H_aeroRef_aeroBody = H_aeroRef_aeroBody.dot( tf.euler_matrix(0, 0, heading_north_yaw, 'sxyz'))

#                # Show debug messages here
#                if debug_enable == 1:
#                    os.system('clear') # This helps in displaying the messages to be more readable
#                    print("DEBUG: Raw RPY[deg]: {}".format( np.array( tf.euler_from_matrix( H_T265Ref_T265body, 'sxyz')) * 180 / m.pi))
#                    print("DEBUG: NED RPY[deg]: {}".format( np.array( tf.euler_from_matrix( H_aeroRef_aeroBody, 'sxyz')) * 180 / m.pi))
#                    print("DEBUG: Raw pos xyz : {}".format( np.array( [data.translation.x, data.translation.y, data.translation.z])))
#                    print("DEBUG: NED pos xyz : {}".format( np.array( tf.translation_from_matrix( H_aeroRef_aeroBody))))

#except KeyboardInterrupt:
#    send_msg_to_gcs('Closing the script...')  

#except:
#    send_msg_to_gcs('ERROR: Camera disconnected')  

#finally:
#    pipe.stop()
#    vehicle.close()
#    print("INFO: Realsense pipeline and vehicle object closed.")
#    sys.exit()
