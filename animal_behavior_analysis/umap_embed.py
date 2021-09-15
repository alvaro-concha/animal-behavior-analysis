"""Produce UMAP embeddings using joint angles and wavelet spectra."""
import numpy as np
from umap import UMAP
from sklearn.decomposition import IncrementalPCA
from utilities import read_pickle, write_pickle
import config


def get_pca_fit_wavelet_spectra(dep_pickle_paths, target_pickle_path):
    """Fit incremental PCA, with whitening, to wavelet spectra features."""
    wavs = np.concatenate(
        [
            read_pickle(path)[:: config.EMB_WAV_SUBSAMPLE_EVERY]
            for path in dep_pickle_paths["wav"]
        ],
    )
    del dep_pickle_paths["wav"]
    pca = IncrementalPCA(whiten=True, copy=False).fit(wavs)
    del wavs
    write_pickle(pca, target_pickle_path)


def get_umap_embeddings(dep_pickle_paths, target_pickle_paths):
    """Produce UMAP embeddings using joint angles and wavelet spectra."""
    angs = np.concatenate(
        [
            read_pickle(path)[:: config.EMB_ANG_SUBSAMPLE_EVERY]
            for path in dep_pickle_paths["ang"]
        ]
    )
    (_, mean, var) = read_pickle(dep_pickle_paths["stat"])
    angs = (angs - mean) / np.sqrt(var)
    write_pickle(angs, target_pickle_paths["ang_sample"])
    del dep_pickle_paths["ang"], dep_pickle_paths["stat"], _, mean, var
    embedding = UMAP(
        metric=config.EMB_METRIC,
        a=config.EMB_A,
        b=config.EMB_B,
        n_neighbors=config.EMB_ANG_N_NEIGHBORS,
        verbose=True,
    ).fit(angs)
    write_pickle(embedding, target_pickle_paths["emb_ang"])
    write_pickle(embedding.embedding_, target_pickle_paths["out_ang"])
    del angs, target_pickle_paths["emb_ang"], embedding

    pca = read_pickle(dep_pickle_paths["pca"])
    wavs = np.concatenate(
        [
            pca.transform(read_pickle(path)[:: config.EMB_WAV_SUBSAMPLE_EVERY])[
                :, : config.EMB_WAV_PCA_NUM_COMPONENTS
            ]
            for path in dep_pickle_paths["wav"]
        ]
    )
    write_pickle(wavs, target_pickle_paths["wav_sample"])
    del dep_pickle_paths["wav"], dep_pickle_paths["pca"], pca
    embedding = UMAP(
        metric=config.EMB_METRIC,
        a=config.EMB_A,
        b=config.EMB_B,
        n_neighbors=config.EMB_WAV_N_NEIGHBORS,
        verbose=True,
    ).fit(wavs)
    write_pickle(embedding, target_pickle_paths["emb_wav"])
    write_pickle(embedding.embedding_, target_pickle_paths["out_wav"])
    del wavs, target_pickle_paths["emb_wav"], embedding


def get_umap_angles_outcomes(emb_angs, mean, var, path_in, path_out):
    """UMAP outcomes from tranined embeddings, using angles as features."""
    angs = read_pickle(path_in)
    angs = (angs - mean) / np.sqrt(var)
    out_angs = emb_angs.transform(angs)
    write_pickle(out_angs, path_out)


def get_umap_wavelets_outcomes(emb_wavs, pca, path_in, path_out):
    """UMAP outcomes from tranined embeddings, using wavelets as features."""
    wavs = read_pickle(path_in)
    wavs = pca.transform(wavs)
    out_wavs = emb_wavs.transform(wavs)
    write_pickle(out_wavs, path_out)
