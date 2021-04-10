import cv2
import numpy as np
import os
from scipy import optimize
from LaneDetection.ImageUtils import processImage
from LaneDetection.ImageUtils import perspectiveWarp
from LaneDetection.LaneLines import plotHistogram
from LaneDetection.LaneLines import slide_window_search
from LaneDetection.LaneLines import general_search
from LaneDetection.LaneLines import measure_lane_curvature
from LaneDetection.LaneLines import draw_lane_lines
from LaneDetection.LaneDeviation import offCenter
from LaneDetection.LaneDeviation import addText



def detect_lanes(image_frame):
    
    frame = image_frame
    birdView, birdViewL, birdViewR, minverse = perspectiveWarp(frame)

    img, hls, grayscale, thresh, blur, canny = processImage(birdView)
    imgL, hlsL, grayscaleL, threshL, blurL, cannyL = processImage(birdViewL)
    imgR, hlsR, grayscaleR, threshR, blurR, cannyR = processImage(birdViewR)

    hist, leftBase, rightBase = plotHistogram(thresh)
    ploty, left_fit, right_fit, left_fitx, right_fitx = slide_window_search(thresh, hist)

    draw_info = general_search(thresh, left_fit, right_fit)
    curveRad, curveDir = measure_lane_curvature(ploty, left_fitx, right_fitx)
    meanPts, result, nwarp = draw_lane_lines(frame, thresh, minverse, draw_info)
    deviation, directionDev = offCenter(meanPts, frame)

    finalImg = addText(result, curveRad, curveDir, deviation, directionDev)
    return finalImg