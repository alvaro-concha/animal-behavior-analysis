"""General data management functions.

Create folders, write and read pickles, and get file paths.
"""
import os
import pickle5
from glob import glob
from errno import EEXIST
from itertools import product

################################ FUNCTIONS #################################

def try_create_folder(folder):
	"""
	Tries to create folder if it doesn't exist already
	Parameters
	----------
	folder: str
		Name of the folder to try to create
	"""
	try:
		os.makedirs(folder)
	except OSError as exc:
		if exc.errno == EEXIST and os.path.isdir(folder):
			pass
		else: raise

def write_pickle(obj, file, folder=None):
	"""
	Writes an object as a pickle file
	Parameters
	----------
	file: str
		Name of the pickle to be created
	folder: str
		Name of the folder to save the pickle in
	obj:
		Object to be pickled
	"""
	if folder != None:
		try_create_folder(folder)
		pickle_path = folder + "/" + file
	else: pickle_path = file

	with open(pickle_path, "wb") as file:
		pickle5.dump(
			obj,
			file,
			protocol=-1,
			fix_imports=False
		)

def read_pickle(file, folder=None):
	"""
	Loads an object from a pickle file
	Parameters
	----------
	file: str
		Name of the pickled file
	folder: str
		Name of the folder containing the pickled file
	Returns
	-------
	object:
		Unpickled object
	"""
	if folder != None:
		pickle_path = folder + "/" + file
	else: pickle_path = file

	with open(pickle_path,"rb") as pickled_object:
		return pickle5.load(pickled_object)

def get_file_paths(path, extension="*.csv"):
	"""
	Parameters
	----------
	path: str
		Desired path in the file system, starting from root
	extension: str
		Extension of the target files in the path
	Returns
	-------
	files: list
		List of files found inside path and its subdirectories
	"""
	files = [
		file 
		for path, subdir, files
		in os.walk(path)
		for file
		in glob(os.path.join(path, extension))
	]
	return files

def get_file_paths_dict(
	file_paths,
	mouse_list, day_list, trial_list, cam_list,
	mouse_kw_fun, day_kw_fun, trial_kw_fun, cam_kw_fun,
	save_pickle=True,
	pickle_name="file_paths_dict.pickle",
	save_folder="Data/Metadata"
):
	"""
	Saves ordered file paths as a pickled dict
	file_paths_dict[(mouse, day, trial, cam)] = file_path
	-----------------------------------------------------
	Parameters
	----------
	file_paths: list
		List of unordered file paths
	{mouse, day, trial, cam}_list: list
		List of mouse IDs, day, trial or camera numbers
	{mouse, day, trial, cam}_kw_fun: function(str) -> str
		Function for mouse ID, day, trial or camera number keyword lookup
	save_pickle: bool
		If true, saves file names as file_paths.pickle
	pickle_name: str
		Name of pickled file to save
	save_folder: str
		Name of folder to save pickle into
	Returns
	-------
	file_paths_dict: dict
		Ordered file paths
	"""
	file_paths_dict = {}
	for mouse, day, trial, cam in product(
		mouse_list, day_list, trial_list, cam_list 
	):
		for file in file_paths:
			if all(
				keyword in file for keyword in [
					mouse_kw_fun(mouse), day_kw_fun(day),
					trial_kw_fun(trial), cam_kw_fun(cam)
				]
			):
				file_paths_dict[(mouse, day, trial, cam)] = file
	if save_pickle:
		write_pickle(file_paths_dict, pickle_name, save_folder)

	return file_paths_dict

def main_data_handling():
	"""
	Define mouse IDs, day, trial and camera numbers. Then, get file paths.
	"""
	mouse_list = [295, 297, 298, 329, 330]
	day_list = [1, 5]
	trial_list = [1, 2, 3, 4, 5]
	cam_list = [1, 2]

	learning_address = "/home/alvaro/Documents/IB/Semestre 7/TM/Files"
	learning_file_paths = get_file_paths(
		learning_address + "/Spreadsheets",
		extension="*.csv"
	)
	learning_file_paths_dict = get_file_paths_dict(
		file_paths=learning_file_paths,
		mouse_list=mouse_list,
		day_list=day_list,
		trial_list=trial_list,
		cam_list=cam_list,
		mouse_kw_fun=lambda mouse: f"ID{mouse}",
		day_kw_fun=lambda day: f"Dia{day}",
		trial_kw_fun=lambda trial: f"trial{trial}",
		cam_kw_fun=lambda cam: f"{cam}DLC",
		save_pickle=True,
		pickle_name="learning_file_paths_dict.pickle",
		save_folder="Data/Metadata"
	)

if __name__ == "__main__":
	main_data_handling()