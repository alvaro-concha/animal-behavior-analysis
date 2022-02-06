"""Mouse behavior analysis modeules configuration parameters.

Control and modify parameters in the modules of the pipeline.

Joint Angles
------------
ANG_MARKER_IDX : array_like
    Array of thruples of marker indices that define an angle

Wavelet Spectra
---------------
WAV_F_SAMPLING : float
    Sampling frequency of motion tracking, in Hz
WAV_F_MIN : float
    Minimum frequency channel value, in Hz
WAV_F_MAX : float
    Maximum frequency channel value, in Hz. Set to Nyquist frequency
WAV_NUM_CHANNELS_{LOW, MID, HIGH} : int
    Number of frequency channels in low, mid and high frequency bands
WAV_F_CHANNELS : array_like
    Step-wise generated, dyadically spaced total frequency channels
WAV_NUM_CHANNELS : int
    Number of total frequency channels
WAV_OMEGA_0 : int
    Morlet wavelet omega_0 parameter. Related to number of observed cycles
WAV_DT : float
    Wavelet transform time step value. Set to inverse of sampling frequency

PCA
---
PCA_WAV_NUM_COMPONENTS : int
    Number of PCA components of wavelet spectra to keep

UMAP Embedding
--------------
EMB_METRIC : str
    Type of metric to use for UMAP embedding
EMB_A : float
    UMAP embedding kernel parameter a
EMB_B : float
    UMAP embedding kernel parameter b
EMB_WAV_N_NEIGHBORS : int
    Number of neighbors to use for UMAP wavelet embedding
EMB_WAV_SUBSAMPLE_EVERY : int
    Number of frames to subsample between for UMAP wavelet embedding
EMB_STP_N_NEIGHBORS : int
    Number of neighbors to use for UMAP steps and poses embedding

KNN Regressor
-------------
KNN_N_NEIGHBORS : int
    Number of neighbors to use for KNN regression
KNN_WEIGHTS : str
    Type of weights to use for KNN regression
KNN_ALGORITHM : str
    Type of algorithm to use for KNN regression
KNN_LEAF_SIZE : int
    Leaf size to use for KNN regression
KNN_N_JOBS : int or -1
    Number of jobs to use for KNN regression

Bundled Edges
-------------
EDG_SUBSAMPLE_EVERY : int
    Number of frames to subsample between for bundled edges

Local Dimension
---------------
LOC_DIM_SUBSAMPLE_EVERY : int
    Number of frames to subsample between for local dimension
LOC_DIM_VARIANCE_THRESHOLD : float
    Threshold (between 0 and 1) of variance to use for local dimension

Watershed Segmentation
----------------------
WAT_RESOLUTION : int
    Resolution for fastKDE probability density estimation
WAT_RADIUS_DENOISE : int
    Radius for the disk-shaped median denoising filter mask
WAT_RADIUS_GRADIENT : int
    Radius for the disk-shaped gradient filter mask
WAT_RADIUS_FOOTPRINT : int
    Radius for the disk-shaped mask to search local maxima within
WAT_CUT_THRESHOLD : float
    8-bit intensity values below these threshold are set to 0
WAT_EMB_WAV_SIGMA : float
    Bandwith for the KDE in the UMAP wavelet embedding
WAT_EMB_WAV_LIM : tuple
    Lower and upper limits in the UMAP wavelet embedding
WAT_EMB_WAV_EDG : tuple
    Smaller lower and upper limits in the UMAP wavelet embedding
WAT_EMB_WAV_MAIN : tuple
    Smallest lower and upper limits in the UMAP wavelet embedding
WAT_EMB_WAV_DIFF_MAIN : float
    Scale of the UMAP wavelet embedding
WAT_EMB_STP_SIGMA : float
    Bandwith for the KDE in the UMAP steps and poses embedding
WAT_EMB_STP_LIM : tuple
    Lower and upper limits in the UMAP steps and poses embedding
WAT_EMB_STP_EDG : tuple
    Smaller lower and upper limits in the UMAP steps and poses embedding
WAT_EMB_STP_MAIN : tuple
    Smallest lower and upper limits in the UMAP steps and poses embedding
WAT_EMB_STP_DIFF_MAIN : float
    Scale of the UMAP steps and poses embedding
LAB_STP_CHANGE_DICT : dict
    Dictionary used to combine certain steps and poses labels
"""
import numpy as np
from dyadic_frequencies import get_dyadic_frequencies

############################### Motion Tracking ################################

IDX_MARKER_DICT = {
    0: "left hind paw",
    1: "right hind paw",
    2: "base tail",
    3: "middle tail",
    4: "back",
    5: "left back",
    6: "right back",
    7: "left front paw",
    8: "right front paw",
    9: "nose",
    10: "left ear",
    11: "right ear",
    12: "top right",
    13: "top left",
    14: "bottom right",
    15: "bottom left",
    16: "smoothed nose",
    17: "center of mass",
}
FRONT_MARKER_IDX = np.arange(7, 12)
BACK_MARKER_IDX = np.arange(7)
BODY_MARKER_IDX = np.arange(12)
CORNER_MARKER_IDX = np.arange(12, 16)
EXTRA_MARKER_IDX = np.arange(16, 18)
CM_MARKER_IDX = [2, 4, 5, 6]
NOSE_MARKER_IDX = [9]
ROTAROD_WIDTH = 57.0  # in mm
ROTAROD_HEIGHT = 30.0  # in mm
CORNER_DESTINATION = np.array(
    [
        [ROTAROD_WIDTH, ROTAROD_HEIGHT],
        [0.0, ROTAROD_HEIGHT],
        [ROTAROD_WIDTH, 0.0],
        [0.0, 0.0],
    ]
)

################################ Joint Angles ##################################

ANG_MARKER_IDX = [
    [4, 17, 0],  # left hindpaw
    [17, 0, 4],  # left hindpaw
    [4, 17, 1],  # right hindpaw
    [17, 1, 4],  # right hindpaw
    [0, 17, 1],  # cm hindpaws
    [17, 2, 3],  # tail
    [2, 17, 4],  # back
    [17, 4, 2],  # back
    [4, 16, 17],  # nose
    [17, 4, 16],  # nose
]

############################### Wavelet Spectra ################################

WAV_F_SAMPLING = 100.0
WAV_F_MIN = 0.1
WAV_F_MAX = 10.0
WAV_NUM_CHANNELS = 50
WAV_F_CHANNELS = get_dyadic_frequencies(WAV_F_MIN, WAV_F_MAX, WAV_NUM_CHANNELS)
WAV_OMEGA_0 = 10.0
WAV_DT = 1.0 / WAV_F_SAMPLING

##################################### PCA ######################################

PCA_WAV_NUM_COMPONENTS = 50

############################### UMAP Embedding #################################

EMB_METRIC = "cosine"
EMB_A = 1.0
EMB_B = 0.4
EMB_WAV_N_NEIGHBORS = 100
EMB_WAV_SUBSAMPLE_EVERY = 2
EMB_STP_N_NEIGHBORS = 30

################################ KNN Regressor #################################

KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = "distance"
KNN_ALGORITHM = "ball_tree"
KNN_LEAF_SIZE = 100
KNN_N_JOBS = -1

################################ Bundled Edges #################################

EDG_SUBSAMPLE_EVERY = 10  # 200k edges

############################### Local Dimension ################################

LOC_DIM_SUBSAMPLE_EVERY = 1
LOC_DIM_VARIANCE_THRESHOLD = 0.8

########################### Watershed Segmentation #############################

WAT_RESOLUTION = 2048
WAT_RADIUS_DENOISE = 2
WAT_RADIUS_GRADIENT = 10
WAT_CUT_THRESHOLD = 75
WAT_RADIUS_FOOTPRINT = 30

WAT_EMB_WAV_SIGMA = 2.0
WAT_EMB_WAV_LIM = (-25.0, 35.0)
WAT_EMB_WAV_EDG = (-21.0, 31.0)
WAT_EMB_WAV_MAIN = (-18, 28.0)
WAT_EMB_WAV_DIFF_MAIN = WAT_EMB_WAV_MAIN[1] - WAT_EMB_WAV_MAIN[0]

WAT_EMB_STP_SIGMA = 1.0
WAT_EMB_STP_LIM = (-15.0, 25.0)
WAT_EMB_STP_EDG = (-15.0, 25.0)
WAT_EMB_STP_MAIN = (-15.0, 25.0)
WAT_EMB_STP_DIFF_MAIN = WAT_EMB_STP_MAIN[1] - WAT_EMB_STP_MAIN[0]

LAB_STP_CHANGE_DICT = {
    1: 1,
    5: 1,
    2: 2,
    3: 3,
    4: 3,
    6: 3,
    7: 5,
    11: 5,
    8: 4,
    9: 4,
    10: 6,
    12: 6,
    14: 7,
    15: 7,
    13: 8,
    16: 9,
    17: 10,
}
