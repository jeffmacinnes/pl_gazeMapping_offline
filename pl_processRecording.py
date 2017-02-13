"""
Process a Pupil Labs gaze recording
-jjm35
"""

# python 2/3 compatibility
from __future__ import division
from __future__ import print_function

import os, sys
from os.path import join
from bisect import bisect_left
import cv2
import numpy as np 
import pandas as pd 
import seaborn as sns
import argparse

# data formatting tools
from gazeDataFormatting import formatGazeData, writeGazeData_csv

# custom pupil-lab projection tools
import gazeMappingTools as gm


"""
This script can be called to process recordings by pupil-labs glasses

Usage:
	python pl_processRecording.py <path to input dir>

Inputs:
	- recording directory: path to recording session directory (typically ../recordings/<date>/<number>)

Outputs:
	- all outputs written to "processed" directory created within input directory
	- movies:
		- orig + gaze overlay
		- orig + evolving heatmap overlay
		- ref + evolving heatmap overlay
	- data:
		- camGazePosition: camera locations
		- camGazePosition_smooth: spike removal and smoothing

Note about coordinates:
The operations below focus on translating between multiple different coordinate systems.
For clarity, here are those different systems and the labels used when referencing each:
	- Camera Frame (frame):		Coordinate system used on frames take from the camera (e.g. scene camera on glasses)
								This is a 2D system with the origin in the top-left corner of the frame, units in pixels
	- Reference Image (ref):	Coordinate system of the reference image that is expected to be found in the camera frame. 
								This is a 2D system with the origin in the top-left corner, units in pixels
	- Object (obj):				Coordinate system of the reference image AS IT OCCURS IN THE PHYSICAL WORLD. This is a 3D
								system, however all z-values are set to 0, defining the plane of the wall that the image is 
								on. The units are in whatever physical coordinates you desire (e.g. inches); whatever unit you 
								set will be used on subsequent measurements, such as camera position. 
"""

def processRecording(input_dir):
	"""
	Open the recording in the specified input dir. 
	Format the pupil data. 
	Loop through each frame of the recording and create output videos
	"""

	# specify output dir (create if necessary)
	outputDir = join(input_dir, 'processed')
	if not os.path.isdir(outputDir):
		os.makedirs(outputDir)

	# format pupil data
	gaze_data = formatGazeData(input_dir)
	
	# write the gaze data to a csv file
	writeGazeData_csv(input_dir, gaze_data)

	# test = open(join(outputDir, 'gazeData.pkl'), 'wb')
	# pickle.dump(gaze_data, test)
	# test.close()










if __name__ == '__main__':
	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('inputDir', help="path to pupil labs recording dir")
	args = parser.parse_args()

	# check if valid dir
	if not os.path.isdir(args.inputDir):
		print('Invalid input dir: ' + args.inputDir)
		sys.exit()
	else:
		processRecording(args.inputDir)



