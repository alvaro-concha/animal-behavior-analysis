"""Produce UMAP embeddings using joint angles and wavelet spectra."""
import multiprocessing as mp
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
    del dep_pickle_paths["ang"], dep_pickle_paths["stat"], _, mean, var
    embedding = UMAP(
        metric=config.EMB_METRIC,
        a=config.EMB_A,
        b=config.EMB_B,
        n_neighbors=config.EMB_ANG_N_NEIGHBORS,
        verbose=True,
    ).fit(angs)
    write_pickle(embedding, target_pickle_paths["emb_ang"])
    del embedding, angs, target_pickle_paths["emb_ang"]

    wavs = np.concatenate(
        [
            read_pickle(path)[:: config.EMB_WAV_SUBSAMPLE_EVERY]
            for path in dep_pickle_paths["wav"]
        ]
    )
    pca = read_pickle(dep_pickle_paths["pca"])
    wavs = pca.transform(wavs)[:, : config.EMB_WAV_PCA_NUM_COMPONENTS]
    del dep_pickle_paths["wav"], dep_pickle_paths["pca"], pca
    embedding = UMAP(
        metric=config.EMB_METRIC,
        a=config.EMB_A,
        b=config.EMB_B,
        n_neighbors=config.EMB_WAV_N_NEIGHBORS,
        verbose=True,
    ).fit(wavs)
    write_pickle(embedding, target_pickle_paths["emb_wav"])
    del embedding, wavs, target_pickle_paths["emb_wav"]


def get_single_umap_outcomes_angles(path_in, path_out, emb_angs, mean, var):
    """Produce UMAP outcomes for a single trial, using angles."""
    print(path_out.stem)
    angs = read_pickle(path_in)
    angs = (angs - mean) / np.sqrt(var)
    out_angs = emb_angs.transform(angs)
    write_pickle(out_angs, path_out)
    del angs, out_angs


def get_parallel_umap_outcomes_angles(
    dep_pickle_paths, target_pickle_paths, emb_angs, mean, var
):
    """Produce UMAP outcomes in parallel, using angles."""
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(
            get_single_umap_outcomes_angles,
            [
                (path_in, path_out, emb_angs, mean, var)
                for path_in, path_out in zip(
                    dep_pickle_paths["ang"], target_pickle_paths["out_ang"]
                )
            ],
        )
        pool.close()
        pool.join()


def get_single_umap_outcomes_wavelets(path_in, path_out, emb_wavs, pca):
    """Produce UMAP outcomes for a single trial, using wavelets."""
    print(path_out.stem)
    wavs = read_pickle(path_in)
    wavs = pca.transform(wavs)
    out_wavs = emb_wavs.transform(wavs)
    write_pickle(out_wavs, path_out)
    del wavs, out_wavs


def get_parallel_umap_outcomes_wavelets(
    dep_pickle_paths, target_pickle_paths, emb_wavs, pca
):
    """Produce UMAP outcomes in parallel, using wavelets."""
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(
            get_single_umap_outcomes_wavelets,
            [
                (path_in, path_out, emb_wavs, pca)
                for path_in, path_out in zip(
                    dep_pickle_paths["wav"], target_pickle_paths["out_wav"]
                )
            ],
        )
        pool.close()
        pool.join()


def get_umap_outcomes(dep_pickle_paths, target_pickle_paths):
    """UMAP outcomes from tranined embeddings using angles and wavelets."""
    emb_angs = read_pickle(dep_pickle_paths["emb_ang"])
    (_, mean, var) = read_pickle(dep_pickle_paths["stat"])
    get_parallel_umap_outcomes_angles(
        dep_pickle_paths, target_pickle_paths, emb_angs, mean, var
    )
    del (
        dep_pickle_paths["emb_ang"],
        emb_angs,
        dep_pickle_paths["stat"],
        _,
        mean,
        var,
        dep_pickle_paths["ang"],
        target_pickle_paths["out_ang"],
    )

    emb_wavs = read_pickle(dep_pickle_paths["emb_wav"])
    pca = read_pickle(dep_pickle_paths["pca"])
    get_parallel_umap_outcomes_wavelets(
        dep_pickle_paths, target_pickle_paths, emb_wavs, pca
    )
    del (
        dep_pickle_paths["emb_wav"],
        emb_wavs,
        dep_pickle_paths["pca"],
        pca,
        dep_pickle_paths["wav"],
        target_pickle_paths["out_wav"],
    )
