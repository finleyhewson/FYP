import argparse
def GetParser():
        parser = argparse.ArgumentParser(description='Reboots vehicle')
        parser.add_argument('--connect',
                            help="Vehicle connection target string. If not specified, a default string will be used.")
        parser.add_argument('--baudrate', type=float,
                            help="Vehicle connection baudrate. If not specified, a default value will be used.")
        parser.add_argument('--vision_position_estimate_msg_hz', type=float,
                            help="Update frequency for VISION_POSITION_ESTIMATE message. If not specified, a default value will be used.")
        parser.add_argument('--scale_calib_enable', default=False, action='store_true',
                            help="Scale calibration. Only run while NOT in flight")
        parser.add_argument('--camera_orientation', type=int,
                            help="Configuration for camera orientation. Currently supported: forward, usb port to the right - 0; downward, usb port to the right - 1, 2: forward tilted down 45deg")
        parser.add_argument('--debug_enable',type=int,
                            help="Enable debug messages on terminal")
        return parser
