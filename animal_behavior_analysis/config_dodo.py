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
MED_PATH = ABS_PATH / "Data/Positions/Median"  # Median filter xys, lhs and perspectives
KAL_PATH = ABS_PATH / "Data/Positions/Kalman"  # Kalman filter xys
QNT_PATH = ABS_PATH / "Data/Positions/Quantile"  # Quantile filter xys
IDX_PATH = ABS_PATH / "Data/Features/Subsample Indices"  # Subsample indices
STP_PATH = ABS_PATH / "Data/Features/Steps and Poses"  # Steps and poses from xys
ANG_PATH = ABS_PATH / "Data/Features/Angles"  # Joint angles from xys
WAV_PATH = ABS_PATH / "Data/Features/Wavelets"  # Wavelet spectra from joint angles
INF_PATH = ABS_PATH / "Data/Features/Information"  # Mutual information and entropy
EMB_PATH = ABS_PATH / "Data/Embeddings/Embedders"  # UMAP embedders
OUT_PATH = ABS_PATH / "Data/Embeddings/Outcomes"  # UMAP outcomes
WAT_PATH = ABS_PATH / "Data/Embeddings/Watershed"  # Watershed segmentation
LAB_PATH = ABS_PATH / "Data/Embeddings/Labels"  # Behavior labels
AGG_PATH = ABS_PATH / "Data/Aggregate"  # Aggregated data for analysis
MTR_PATH = ABS_PATH / "Data/Metrics"  # Metrics (size, performance, aggregate, etc.)
MET_PATH = ABS_PATH / "Data/Metadata"  # Spreadsheet paths
VID_PATH = ABS_PATH / "Data/Videos"  # Rotarod videos
FIG_PATH = ABS_PATH / "Figures"  # Output figures
QUA_PATH = ABS_PATH / "Figures/Quality"  # Quality control
SPR_PATH.mkdir(parents=True, exist_ok=True)
MED_PATH.mkdir(parents=True, exist_ok=True)
KAL_PATH.mkdir(parents=True, exist_ok=True)
QNT_PATH.mkdir(parents=True, exist_ok=True)
IDX_PATH.mkdir(parents=True, exist_ok=True)
STP_PATH.mkdir(parents=True, exist_ok=True)
ANG_PATH.mkdir(parents=True, exist_ok=True)
WAV_PATH.mkdir(parents=True, exist_ok=True)
INF_PATH.mkdir(parents=True, exist_ok=True)
EMB_PATH.mkdir(parents=True, exist_ok=True)
OUT_PATH.mkdir(parents=True, exist_ok=True)
WAT_PATH.mkdir(parents=True, exist_ok=True)
LAB_PATH.mkdir(parents=True, exist_ok=True)
AGG_PATH.mkdir(parents=True, exist_ok=True)
MTR_PATH.mkdir(parents=True, exist_ok=True)
MET_PATH.mkdir(parents=True, exist_ok=True)
VID_PATH.mkdir(parents=True, exist_ok=True)
FIG_PATH.mkdir(parents=True, exist_ok=True)
QUA_PATH.mkdir(parents=True, exist_ok=True)

################################ Subjetct Data #################################

MOUSE_LIST = [262, 263, 264, 265, 282, 295, 297, 298, 329, 330]
HP_MOUSE_LIST = [265, 297, 329]
LP_MOUSE_LIST = [262, 263, 264, 282, 295, 298, 330]
DAY_LIST = [1, 2, 3, 4, 5]
TRIAL_LIST = [1, 2, 3, 4, 5]
CAM_LIST = [1, 2]  # FRONT_CAM = 1, BACK_CAM = 2
PREPROCESS_KEY_LIST = list(product(MOUSE_LIST, DAY_LIST, TRIAL_LIST, CAM_LIST))
PREPROCESS_SUBJECT_NAME = "M{}D{}T{}C{}"
KEY_LIST = list(product(MOUSE_LIST, DAY_LIST, TRIAL_LIST))
SUBJECT_NAME = "M{}D{}T{}"
