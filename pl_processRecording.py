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
import time
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
	framesToUse = np.arange(0, 10, 1)	

	# start time
	process_startTime = time.time()

	# specify output dir (create if necessary)
	outputDir = join(inputDir, 'processed')
	if not os.path.isdir(outputDir):
		os.makedirs(outputDir)

	### Prep the gaze data ################################
	print('Prepping gaze data...')
	# format pupil data
	gazeData_world, frame_timestamps = formatGazeData(inputDir)
	
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


	### Loop through frames of world video #################################
	print('Processing frames....')
	frameProcessing_startTime = time.time()
	frameCounter = 0
	while vid.isOpened():
		# read the next frame of the video
		ret, frame = vid.read()
		
		# check if it's a valid frame
		if (ret==True) and (frameCounter in framesToUse):

			# grab the gazeData (world coords) for this frame only
			thisFrame_gazeData_world = gazeWorld_df.loc[gazeWorld_df['index'] == frameCounter]

			# submit this frame to the processing function
			processedFrame = processFrame(frameCounter, frame, mapper, thisFrame_gazeData_world, frame_timestamps)

			
			# append this frames gaze data file to the bigger list

			# make the heatmaps


			# Write out this frame's different video files
			vidOut_orig.write(processedFrame['origFrame'])
			vidOut_gaze.write(processedFrame['gazeFrame'])



		# increment frame counter
		frameCounter += 1
		if frameCounter > np.max(framesToUse):
			vid.release()
			vidOut_orig.release()
			vidOut_gaze.release()

	endTime = time.time()
	frameProcessing_time = endTime - frameProcessing_startTime
	print('Total time: %s seconds' % frameProcessing_time)
	print('Avg time/frame: %s seconds' % (frameProcessing_time/framesToUse.shape[0]) )


def processFrame(frameCounter, frame, mapper, thisFrame_gazeData_world, frame_timestamps):
	""" Compute all transformations on a given frame """

	fr = {}		# create dict to store info for this frame
	fr['frameNum'] = frameCounter		# store frame number
	
	# create copy of original frame
	origFrame = frame.copy()
	fr['origFrame'] = origFrame 		# store

	# convert to grayscale
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# get the timestamp for this frame
	frame_ts = frame_timestamps[frameCounter]
	fr['frame_ts'] = frame_ts 			# store

	# find the key points and features on this frame
	frame_kp, frame_des = mapper.findFeatures(frame_gray)
	print('found %s features on frame %s' %(len(frame_kp), frameCounter))

	# look for matching keypoints on the reference stimulus
	if len(frame_kp) < 2:
		ref_matchPts = None
	else:
		ref_matchPts, frame_matchPts = mapper.findMatches(frame_kp, frame_des)

	# check if matches were found
	try:
		numMatches = ref_matchPts.shape[0]
		
		# if sufficient number of matches....
		if numMatches > 10:
			print('found %s matches on frame %s' %(numMatches, frameCounter))
			sufficientMatches = True
		else:
			print('Insufficient matches (%s matches) on frame %s' %(numMatches, frameCounter))
			sufficientMatches = False

	except:
		print ('no matches found on frame %s' % frameCounter)
		sufficientMatches = False
		pass

	# Uses matches to find 2D and 3D transformations
	if not sufficientMatches:
		# if not enough matches on this frame, store the untouched frames
		fr['gazeFrame'] = origFrame

	else:
		### 3D operations ##########################
		# get mapping from camera to 3D location of reference image. Reference match points treated as 2D plane in the world (z=0)
		rvec, tvec = mapper.PnP_3Dmapping(ref_matchPts, frame_matchPts)

		# calculate camera position & orientation
		camPosition, camOrientation = mapper.getCameraPosition(rvec, tvec)
		fr['camPosition'] = camPosition
		fr['camOrientation'] = camOrientation

		### 2D operations ###########################
		# get the transformation matrices to map between world frame and reference stimuli
		ref2frame_2D, frame2ref_2D = mapper.get2Dmapping(ref_matchPts, frame_matchPts)

		### Gaze data operations ####################
		if thisFrame_gazeData_world.shape[0] == 0:
			# if no gaze points for this frame
			drawGazePt = False

			# store empty dataframe to store gaze data in frame, reference, and object coordinates
			gazeData_df = pd.DataFrame(columns=['gaze_ts', 'worldFrame', 'confidence',
										'frame_gazeX', 'frame_gazeY',
										'ref_gazeX', 'ref_gazeY', 
										'obj_gazeX', 'obj_gazeY', 'obj_gazeZ'])
			fr['gazeData'] = gazeData_df

		else:
			drawGazePt = True

			# create dataframes to write gaze data into
			gazeData_df = pd.DataFrame(columns=['gaze_ts', 'worldFrame', 'confidence',
										'frame_gazeX', 'frame_gazeY',
										'ref_gazeX', 'ref_gazeY', 
										'obj_gazeX', 'obj_gazeY', 'obj_gazeZ'])
			
			# grab all gaze data for this frame, translate to different coordinate systems
			for i,gazeRow in thisFrame_gazeData_world.iterrows():
				ts = gazeRow['timestamp']
				frameNum = frameCounter
				conf = gazeRow['confidence']

				# translate normalized gaze location to screen coords (note: pupil labs recorded normalized coords, with origin in bottom left)
				frame_gazeX = gazeRow['norm_pos_x'] * frame_gray.shape[1]
				frame_gazeY = frame_gray.shape[0] - (gazeRow['norm_pos_y'] * frame_gray.shape[0])

				# convert coordinates from frame to reference stimulus coordinates
				ref_gazeX, ref_gazeY = mapper.mapCoords2D((frame_gazeX, frame_gazeY), frame2ref_2D)

				# convert from reference stimulus to object coordinates
				objCoords = mapper.ref2obj(np.array([ref_gazeX, ref_gazeY]).reshape(1,2))
				obj_gazeX, obj_gazeY, obj_gazeZ = objCoords.ravel()

				# create dict
				thisRow_df = pd.DataFrame({'gaze_ts': ts, 'worldFrame': frameNum, 'confidence':conf,
											'frame_gazeX': frame_gazeX, 'frame_gazeY': frame_gazeY,
											'ref_gazeX': ref_gazeX, 'ref_gazeY': ref_gazeY,
											'obj_gazeX': obj_gazeX, 'obj_gazeY': obj_gazeY, 'obj_gazeZ': obj_gazeZ},
											index=[i])

				# append this row to the gaze data dataframe
				gazeData_df = pd.concat([gazeData_df, thisRow_df])

			# store gaze data
			fr['gazeData'] = gazeData_df

			# draw circles for gaze locations
			gazeFrame = origFrame.copy()
			for i,row in gazeData_df.iterrows():
				frame_gazeX = int(row['frame_gazeX'])
				frame_gazeY = int(row['frame_gazeY'])

				# set color for last value to be different than previous values for this frame
				if i == gazeData_df.index.max():
					cv2.circle(gazeFrame, (frame_gazeX, frame_gazeY), 10, [96, 52, 234], -1)
				else:
					cv2.circle(gazeFrame, (frame_gazeX, frame_gazeY), 8, [168, 231, 86], -1)

			# store the gaze frame
			fr['gazeFrame'] = gazeFrame

	# Return the dict holding all of the info for this frame
	return fr


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



