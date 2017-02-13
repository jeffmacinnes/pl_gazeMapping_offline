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
	gaze_list = pupil_data[b'gaze_positions']		# gaze position (world camera)

	# load timestamps 
	timestamps_path = join(input_dir, 'world_timestamps.npy')
	timestamps = np.load(timestamps_path)

	# align gaze with world camera timestamps
	gaze_by_frame = correlate_data(gaze_list, timestamps)

	return gaze_by_frame


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

	data.sort(key=lambda d: d[b'timestamp'])

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

		if datum[b'timestamp'] <= ts:
			datum['index'] = frame_idx
			data_by_frame[frame_idx].append(datum)
			data_index +=1
		else:
			frame_idx+=1

	return data_by_frame

def writeGazeData_csv(input_dir, gazeData_dict):
	"""
	after the gaze data has been loaded from the pickle file, this function will write it to 
	a readable csv file within the input_dir
	"""

	csv_file = join(input_dir, 'processed', 'gazeData.csv')
	export_range = slice(0,len(gazeData_dict))
	with open(csv_file, 'w', encoding='utf-8', newline='') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter='\t')
		csv_writer.writerow(("timestamp",
							 "index",
							 "confidence",
							 "norm_pos_x",
							 "norm_pos_y",
							 "base_data",
							 "gaze_point_3d_x",
							 "gaze_point_3d_y",
							 "gaze_point_3d_z",
							 "eye_center0_3d_x",
							 "eye_center0_3d_y",
							 "eye_center0_3d_z",
							 "gaze_normal0_x",
							 "gaze_normal0_y",
							 "gaze_normal0_z",
							 "eye_center1_3d_x",
							 "eye_center1_3d_y",
							 "eye_center1_3d_z",
							 "gaze_normal1_x",
							 "gaze_normal1_y",
							 "gaze_normal1_z"))

		for g in list(chain(*gazeData_dict[export_range])):
			data = ['{}'.format(g[b"timestamp"]), g["index"], g[b"confidence"], g[b"norm_pos"][0], g[b"norm_pos"][1],
					" ".join(['{}-{}'.format(b[b'timestamp'], b[b'id']) for b in g[b'base_data']])]  # use str on timestamp to be consitant with csv lib.

			# add 3d data if avaiblable
			if g.get('gaze_point_3d', None) is not None:
				data_3d = [g[b'gaze_point_3d'][0], g[b'gaze_point_3d'][1], g[b'gaze_point_3d'][2]]

				# binocular
				if g.get('eye_centers_3d' ,None) is not None:
					data_3d += g[b'eye_centers_3d'].get(0, [None, None, None])
					data_3d += g[b'gaze_normals_3d'].get(0, [None, None, None])
					data_3d += g[b'eye_centers_3d'].get(1, [None, None, None])
					data_3d += g[b'gaze_normals_3d'].get(1, [None, None, None])
				# monocular
				elif g.get('eye_center_3d', None) is not None:
					data_3d += g[b'eye_center_3d']
					data_3d += g[b'gaze_normal_3d']
					data_3d += [None]*6
			else:
				data_3d = [None]*15
			data += data_3d
			csv_writer.writerow(data)
