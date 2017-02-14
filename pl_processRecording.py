"""
Process a Pupil Labs gaze recording
-jjm35
"""

# python 2/3 compatibility
from __future__ import division
from __future__ import print_function

import os, sys, shutil
from os.path import join
from bisect import bisect_left
import cv2
import numpy as np 
import pandas as pd 
import seaborn as sns
import argparse

import multiprocessing

# data formatting tools
from gazeDataFormatting import formatGazeData, writeGazeData_world

# custom pupil-lab projection tools
import gazeMappingTools as gm

OPENCV3 = (cv2.__version__.split('.')[0] == '3')
print("OPENCV version " + cv2.__version__)


"""
This script can be called to process recordings by pupil-labs glasses

Usage:
	python pl_processRecording.py <path to input dir> <path to reference stimulus> <path to camera calibration JSON>

Inputs:
	- recording directory: path to recording session directory (typically ../recordings/<date>/<number>)
	- reference stimulus: path to a decently high resolution jpg of the stimulus you are attempting to map gaze on to
	- camera calibration: path to the camera calibration JSON file for this make/model of eye-tracker

Outputs:
	- all outputs written to "processed" directory created within input directory
	- movies (all output movies for the specified frames only):
		- orig
		- orig + gaze overlay
		- orig + evolving heatmap overlay
		- ref + evolving heatmap overlay
	- data:
		- camGazePosition: camera locations
		- camGazePosition_smooth: spike removal and smoothing

Note about coordinates:
The operations below focus on translating between multiple different coordinate systems.
For clarity, here are those different systems and the labels used when referencing each:
	- World Frame (world):		Coordinate system used on frames take from the camera (e.g. world camera on glasses)
								This is a 2D system with the origin in the bottom-left corner of the frame, units in pixels
	- Reference Image (ref):	Coordinate system of the reference image that is expected to be found in the camera frame. 
								This is a 2D system with the origin in the top-left corner, units in pixels
	- Object (obj):				Coordinate system of the reference image AS IT OCCURS IN THE PHYSICAL WORLD. This is a 3D
								system, however all z-values are set to 0, defining the plane of the wall that the image is 
								on. The units are in whatever physical coordinates you desire (e.g. inches); whatever unit you 
								set will be used on subsequent measurements, such as camera position. 
"""

def processRecording(inputDir, refFile, cameraCalib):
	"""
	Open the recording in the specified input dir. 
	Format the pupil data. 
	Loop through each frame of the recording and create output videos
	"""
	# Settings:
	framesToUse = np.arange(0, 200, 1)	

	# specify output dir (create if necessary)
	outputDir = join(inputDir, 'processed')
	if not os.path.isdir(outputDir):
		os.makedirs(outputDir)

	### Prep the gaze data ################################
	print('Prepping gaze data...')
	# format pupil data
	gazeData_world = formatGazeData(inputDir)
	
	# write the gaze data (world camera coords) to a csv file
	writeGazeData_world(inputDir, gazeData_world)

	# read in the csv file as pandas dataframe
	gazeWorld_df = pd.read_table(join(outputDir, 'gazeData_world.csv'))


	### Prep the reference stimulus ########################
	print('Prepping reference stimulus...')
	shutil.copy(refFile, outputDir) 	# put a copy of the reference file in the outputDir
	refStim = cv2.imread(refFile)   		# load in ref stimulus
	
	refStim_dims = (refStim.shape[1], refStim.shape[0])  # pixel dims of stimulus (height, width)
	obj_dims = (30,20) 		# real world dims (height, width) in inches of the stimulus

	# instantiate the gazeMappingTool object
	mapper = gm.GazeMapper(cameraCalib, refStim, obj_dims)


	### Prep the video data ################################
	print('Prepping video data...')
	vid_path = join(inputDir, 'world.mp4')

	# load the video, get parameters
	vid = cv2.VideoCapture(vid_path)
	if OPENCV3:
		totalFrames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
		vidSize = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		fps = vid.get(cv2.CAP_PROP_FPS)
		vidCodec = cv2.VideoWriter_fourcc(*'mp4v')
	else:
		totalFrames = vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
		vidSize = (int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
		fps = vid.get(cv2.cv.CV_CAP_PROP_FPS)
		vidCodec = cv2.cv.CV_FOURCC(*'mp4v')

	# define output videos
	output_prefix = refFile.split('/')[-1:][0].split('.')[0] 	# set the output prefix based on the reference image
	
	vidOutFile_orig = join(outputDir, (output_prefix + '_orig.m4v'))
	vidOut_orig = cv2.VideoWriter()
	vidOut_orig.open(vidOutFile_orig, vidCodec, fps, vidSize, True)	

	vidOutFile_gaze = join(outputDir, (output_prefix + '_gaze.m4v'))
	vidOut_gaze = cv2.VideoWriter()
	vidOut_gaze.open(vidOutFile_gaze, vidCodec, fps, vidSize, True)

	vidOutFile_heatmap = join(outputDir, (output_prefix + '_heatmapEvolving.m4v'))
	vidOut_heatmap = cv2.VideoWriter()
	vidOut_heatmap.open(vidOutFile_heatmap, vidCodec, fps, vidSize, True)

	vidOutFile_refHeatmap = join(outputDir, (output_prefix + 'Ref_heatmapEvolving.m4v'))
	vidOut_refHeatmap = cv2.VideoWriter()
	vidOut_refHeatmap.open(vidOutFile_refHeatmap, vidCodec, fps, refStim_dims, True)


	### Set up parellelization for video frames #############################
	print('Setting up multiprocessing for tasks...')
	n_cpu = multiprocessing.cpu_count()
	print("Using %i cores" % n_cpu)
	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	
	# build list of tasks
	task_args = []
	frameCounter = 0
	while vid.isOpened():
		# read the next frame of the video
		ret, frame = vid.read()
		
		# check if it's a valid frame
		if (ret==True) and (frameCounter in framesToUse):

			# add this frame to the tasks
			task_args.append((frameCounter, frameCounter))

		# increment frame counter
		frameCounter += 1
		if frameCounter > np.max(framesToUse):
			vid.release()

	### Execute tasks in parallel ############################################
	print('Submitting tasks to pool...')
	results = []
	for t in task_args:
		results.append(pool.apply_async(processFrame, t))

	for r in results:
		i,c = r.get()
		print(i)
		#print(type(r))

def processFrame(frameCounter, blah):
	i = frameCounter
	c = blah
	return (i, c)




if __name__ == '__main__':
	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('inputDir', help="path to pupil labs recording dir")
	parser.add_argument('referenceFile', help="path to reference stimuli")
	parser.add_argument('cameraCalibration', help="path to camera calibration file")
	args = parser.parse_args()

	# check if valid dir
	if not os.path.isdir(args.inputDir):
		print('Invalid input dir: ' + args.inputDir)
		sys.exit()
	else:
		processRecording(args.inputDir, args.referenceFile, args.cameraCalibration)



