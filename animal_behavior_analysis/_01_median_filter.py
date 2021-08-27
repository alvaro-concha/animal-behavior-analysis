"""General data management functions.

Create folders, write and read pickles, and get file paths.
"""
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from _00_data_handling import (
	read_pickle, write_pickle, get_file_paths, get_file_paths_dict
)

################################## FUNCTIONS ###################################

def main_median_filter():
	"""
	Load X-Y coords and likelihoods into dict with keys (mouse, day, trial, cam).
	Apply median filter to coords and likelihoods.
	Then, reshape the coordinates.
	"""
	example_mouse_list = [295, 330]
	example_day_list = [1, 5]
	example_trial_list = [1, 5]
	example_cam_list = [1, 2]
	median_window = 11

	learning_address = "/home/alvaro/Documents/IB/Semestre 7/TM/Files"
	learning_file_paths = get_file_paths(
		learning_address + "/Spreadsheets",
		extension="*.csv"
	)
	learning_file_paths_dict = get_file_paths_dict(
		file_paths=learning_file_paths,
		mouse_list=example_mouse_list,
		day_list=example_day_list,
		trial_list=example_trial_list,
		cam_list=example_cam_list,
		mouse_kw_fun=lambda mouse: f"ID{mouse}",
		day_kw_fun=lambda day: f"Dia{day}",
		trial_kw_fun=lambda trial: f"trial{trial}",
		cam_kw_fun=lambda cam: f"{cam}DLC",
		save_pickle=True,
		pickle_name="learning_file_paths_dict.pickle",
		save_folder="Data/Metadata"
	)

	learning_xys = {}
	learning_lhs = {}
	for key, path in learning_file_paths_dict.items():
		print(key)
		learning_xys[key] = pd.read_csv(
			path,
			skiprows=2,
			usecols=np.concatenate([[i, i + 1] for i in range(1, 49, 3)]),
		).dropna(how="all").values
		learning_lhs[key] = pd.read_csv(
			path,
			skiprows=2,
			usecols=range(3, 49, 3)
		).dropna(how="all").values

	for key in learning_file_paths_dict.keys():
		learning_xys[key] = medfilt(
			learning_xys[key], kernel_size=(median_window, 1)
		).reshape((learning_xys[key].shape[0], -1, 2))
		learning_lhs[key] = medfilt(
			learning_lhs[key], kernel_size=(median_window, 1)
		)

##################################### MAIN #####################################

if __name__ == "__main__":
	main_median_filter()