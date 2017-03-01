"""
Tools for formatting the gaze data from a pupil-labs recording
"""
import os
import csv
from os.path import join
from itertools import chain
import numpy as np

# tools for loading pickled pupil data
try:
	import cPickle as pickle
except ImportError:
	import pickle


def formatGazeData(input_dir):
	"""
	- from the recording dir, load the pupil_data and timestamps.
	- get the "gaze" data from pupil_data (i.e. the gaze location w/r/t world camera
	- sync gaze data with the world_timestamps array

	"""
	# load pupil data
	pupil_data_path = join(input_dir, 'pupil_data')
	try: 
		with open(pupil_data_path, 'rb') as fh:
			pupil_data = pickle.load(fh, encoding='bytes')
	except pickle.UnpicklingError:
		raise ValueError
	gaze_list = pupil_data['gaze_positions']		# gaze position (world camera)

	# load timestamps 
	timestamps_path = join(input_dir, 'world_timestamps.npy')
	frame_timestamps = np.load(timestamps_path)

	# align gaze with world camera timestamps
	gaze_by_frame = correlate_data(gaze_list, frame_timestamps)

	return gaze_by_frame, frame_timestamps


def getCameraCalibration(input_dir):
	"""
	load the camera calibration file that gets stored with pupil labs recording
	"""
	# load calib pickle
	calib_path = join(input_dir, 'camera_calibration')
	try:
		with open(calib_path, 'rb') as fh:
			calib_data = pickle.load(fh, encoding='bytes')
	except pickle.UnpicklingError:
		raise ValueError

	return calib_data


def correlate_data(data,timestamps):
	'''
	data:  list of data :
		each datum is a dict with at least:
			timestamp: float

	timestamps: timestamps list to correlate  data to

	this takes a data list and a timestamps list and makes a new list
	with the length of the number of timestamps.
	Each slot contains a list that will have 0, 1 or more assosiated data points.

	Finally we add an index field to the datum with the associated index
	'''
	timestamps = list(timestamps)
	data_by_frame = [[] for i in timestamps]

	frame_idx = 0
	data_index = 0

	data.sort(key=lambda d: d['timestamp'])

	while True:
		try:
			datum = data[data_index]
			# we can take the midpoint between two frames in time: More appropriate for SW timestamps
			ts = ( timestamps[frame_idx]+timestamps[frame_idx+1] ) / 2.
			# or the time of the next frame: More appropriate for Sart Of Exposure Timestamps (HW timestamps).
			# ts = timestamps[frame_idx+1]
		except IndexError:
			# we might loose a data point at the end but we dont care
			break

		if datum['timestamp'] <= ts:
			datum['frame_idx'] = frame_idx
			data_by_frame[frame_idx].append(datum)
			data_index +=1
		else:
			frame_idx+=1

	return data_by_frame

def writeGazeData_world(input_dir, gazeData_dict):
	"""
	after the gaze data has been loaded from the pickle file, this function will write it to 
	a readable csv file within the input_dir
	"""

	csv_file = join(input_dir, 'processed', 'gazeData_world.csv')
	export_range = slice(0,len(gazeData_dict))
	with open(csv_file, 'w', encoding='utf-8', newline='') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter='\t')
		csv_writer.writerow(("timestamp",
							 "frame_idx",
							 "confidence",
							 "norm_pos_x",
							 "norm_pos_y"))

		for g in list(chain(*gazeData_dict[export_range])):
			data = ['{}'.format(g["timestamp"]), g["frame_idx"], g["confidence"], g["norm_pos"][0], g["norm_pos"][1]]  # use str on timestamp to be consitant with csv lib.

			csv_writer.writerow(data)
