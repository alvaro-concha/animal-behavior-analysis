"""Pickle sample wavelet PCA, features of sample UMAP embedding."""
import numpy as np
import config_dodo
import config
from utilities import read_pickle, write_pickle

dep_pickle_paths = {
    "wav": [],
    "pca": config_dodo.WAV_PATH / "pca_fit_wav.pickle",
}
for key in config_dodo.KEY_LIST:
    name = config_dodo.SUBJECT_NAME.format(*key)
    pickle_end = name + ".pickle"
    dep_pickle_paths["wav"].append(config_dodo.WAV_PATH / f"wav_{pickle_end}")
pca = read_pickle(dep_pickle_paths["pca"])
wavs = np.concatenate(
    [
        pca.transform(read_pickle(path)[:: config.EMB_WAV_SUBSAMPLE_EVERY])[
            :, : config.EMB_WAV_PCA_NUM_COMPONENTS
        ]
        for path in dep_pickle_paths["wav"]
    ]
)
write_pickle(wavs, config_dodo.WAV_PATH / "wav_sample.pickle")
