import cv2
import numpy as np
import pyrealsense2 as rs

from landmarks_detector import gesture_points_detector

pipe = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)

pipe.start(cfg)

while True:
    " get image from Realsence camera "
    frame = pipe.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = cv2.flip(depth_image, 1)
    color_image = cv2.flip(color_image, 1)

    frame = color_image
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    points, frame, gesture_ml = gesture_points_detector(color_image, frame)

    cv2.imshow('gesture_recognition', frame)
    cv2.imshow('depth', depth_image)
    # cv2.imshow('rgb', color_image)

    if cv2.waitKey(1) == ord('q'):
        break

pipe.stop()