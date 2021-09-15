"""Mouse behavior preprocessing pipeline configuration parameters.

Control and modify parameters used in the dodo.py pipeline.

File System
-----------
{X}_PATH : list of pathlib.Path
    Path to a specific folder(X)

Data Handling
-------------
{MOUSE, DAY, TRIAL, CAM}_LIST : list of int
    List of mouse IDs, day, trial or camera numbers
PREPROCESS_KEY_LIST : list of tuple
    List of keys at the beginning of preprocessing, separated cameras
KEY_LIST : list of tuple
    List of keys with front and back cameras combined
PREPROCESS_SUBJECT_NAME : str
    Formatable string: name of subjects, separated cameras
SUBJECT_NAME : str
    Formatable string: name of subjects with front and back cameras combined
"""
from pathlib import Path
from itertools import product

################################# File System ##################################

ABS_PATH = Path(__file__).parent.parent.parent.absolute()
SPR_PATH = ABS_PATH / "Data/Spreadsheets"  # Original source spreadsheets
MED_PATH = ABS_PATH / "Data/Positions/Median"  # xys, lhs and perspective matrices
KAL_PATH = ABS_PATH / "Data/Positions/Kalman"  # xys
ANG_PATH = ABS_PATH / "Data/Features/Angles"  # Joint angles from xys
WAV_PATH = ABS_PATH / "Data/Features/Wavelets"  # Wavelet spectra from joint angles
EMB_PATH = ABS_PATH / "Data/Ethograms/Embeddings"  # UMAP embeddings
OUT_PATH = ABS_PATH / "Data/Ethograms/Outcomes"  # UMAP outcomes
CLU_PATH = ABS_PATH / "Data/Ethograms/Clusterers"  # HDBSCAN clusterers
LAB_PATH = ABS_PATH / "Data/Ethograms/Labels"  # Cluster labels
AGG_PATH = ABS_PATH / "Data/Aggregate"  # Aggregated data: labels, subjects, etc
MET_PATH = ABS_PATH / "Data/Metadata"  # Spreadsheet paths
VID_PATH = ABS_PATH / "Data/Videos"  # Rotarod videos
ANI_PATH = ABS_PATH / "Animations"  # Output animations
FIG_PATH = ABS_PATH / "Figures"  # Output figures

################################ Subjetct Data #################################

MOUSE_LIST = [295, 297, 298, 329, 330]
DAY_LIST = [1, 2, 3, 4, 5]
TRIAL_LIST = [1, 2, 3, 4, 5]
CAM_LIST = [1, 2]  # FRONT_CAM = 1, BACK_CAM = 2
PREPROCESS_KEY_LIST = list(product(MOUSE_LIST, DAY_LIST, TRIAL_LIST, CAM_LIST))
PREPROCESS_SUBJECT_NAME = "M{}D{}T{}C{}"
KEY_LIST = list(product(MOUSE_LIST, DAY_LIST, TRIAL_LIST))
SUBJECT_NAME = "M{}D{}T{}"
