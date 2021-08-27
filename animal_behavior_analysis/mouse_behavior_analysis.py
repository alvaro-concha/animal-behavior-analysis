"""Mouse behavior analysis pipeline.

Load data, transform it and produce figures and animations.
"""
import mouse_behavior_analysis_parameters as p
from _00_data_handling import (
	read_pickle, write_pickle, get_file_paths, get_file_paths_dict
)

##################################### MAIN #####################################

if __name__ == "__main__":
	# print(help(p))
	# p.mouse_list = [295, 297, 298, 329, 330]
	# p.day_list = [1, 5]
	# p.trial_list = [1, 2, 3, 4, 5]
	# p.cam_list = [1, 2]

	learning_address = "/home/alvaro/Documents/IB/Semestre 7/TM/Files"
	learning_file_paths = get_file_paths(
		learning_address + "/Spreadsheets",
		extension="*.csv"
	)

	learning_file_paths_dict = get_file_paths_dict(
		file_paths=learning_file_paths,
		mouse_list=p.mouse_list,
		day_list=p.day_list,
		trial_list=p.trial_list,
		cam_list=p.cam_list,
		mouse_kw_fun=lambda mouse: f"ID{mouse}",
		day_kw_fun=lambda day: f"Dia{day}",
		trial_kw_fun=lambda trial: f"trial{trial}",
		cam_kw_fun=lambda cam: f"{cam}DLC",
		save_pickle=True,
		pickle_name="learning_file_paths_dict.pickle",
		save_folder="Data/Metadata"
	)

	print(len(learning_file_paths_dict))