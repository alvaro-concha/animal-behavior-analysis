"""Mouse behavior analysis modeules configuration parameters.

Control and modify parameters in the modules of the pipeline.

Wavelet Spectra
---------------
WAV_OMEGA_0 : int
    Morlet wavelet omega_0 parameter. Related to number of observed cycles

UMAP Embedding
--------------
"""

############################### Wavelet Spectra ################################

WAV_OMEGA_0 = 10.0

################################ UMAP Embedding ################################

EMB_METRIC = "cosine"
EMB_A = 1.0
EMB_B = 0.4
EMB_ANG_N_NEIGHBORS = 100
EMB_ANG_SUBSAMPLE_EVERY = 10
EMB_WAV_PCA_NUM_COMPONENTS = 200
EMB_WAV_OMEGA_FACTOR = 3
EMB_WAV_N_NEIGHBORS = int(10 * EMB_WAV_OMEGA_FACTOR * WAV_OMEGA_0)
EMB_WAV_SUBSAMPLE_EVERY = int(EMB_WAV_OMEGA_FACTOR * WAV_OMEGA_0)
