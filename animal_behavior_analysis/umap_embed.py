"""Produce UMAP embeddings using joint angles and wavelet spectra."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import datashader.bundling as bd
from umap import UMAP
from umap.utils import submatrix
from utilities import read_pickle, write_pickle
import config


def get_umap_wavelets_embedding(dep_pickle_paths, target_pickle_paths):
    """Produce UMAP embeddings using joint angles and wavelet spectra.
    Fit a KNeighborsRegressor to embed new points to the UMAP.
    To emulate cosine distance, normalize the features.
    """
    pca_wavs = read_pickle(dep_pickle_paths["pca_wav"])
    wavs = np.concatenate(
        [
            read_pickle(path_wav)[
                read_pickle(path_idx)[:: config.EMB_WAV_SUBSAMPLE_EVERY]
            ]
            for path_idx, path_wav in zip(
                dep_pickle_paths["idx"], dep_pickle_paths["wav"]
            )
        ]
    )
    write_pickle(wavs, target_pickle_paths["wav_sample"])
    wavs = pca_wavs.transform(wavs)[:, : config.PCA_WAV_NUM_COMPONENTS]
    write_pickle(wavs, target_pickle_paths["pca_wav_sample"])
    del (
        dep_pickle_paths["idx"],
        dep_pickle_paths["wav"],
        dep_pickle_paths["pca_wav"],
        pca_wavs,
    )
    embedding = UMAP(
        metric=config.EMB_METRIC,
        a=config.EMB_A,
        b=config.EMB_B,
        n_neighbors=config.EMB_WAV_N_NEIGHBORS,
        verbose=True,
    ).fit(wavs)
    write_pickle(embedding, target_pickle_paths["emb_wav"])
    write_pickle(embedding.embedding_, target_pickle_paths["out_wav"])
    knn = KNeighborsRegressor(
        n_neighbors=config.KNN_N_NEIGHBORS,
        weights=config.KNN_WEIGHTS,
        algorithm=config.KNN_ALGORITHM,
        leaf_size=config.KNN_LEAF_SIZE,
        n_jobs=config.KNN_N_JOBS,
    ).fit(normalize(wavs), embedding.embedding_)
    write_pickle(knn, target_pickle_paths["knn_wav"])
    del wavs, target_pickle_paths["emb_wav"], embedding


def get_umap_wavelets_outcomes(pca_wavs, knn_wavs, path_in, path_out):
    """UMAP outcomes from tranined KNN, using wavelets as features.
    To emulate cosine distance in KNN, normalize the features.
    """
    wavs = read_pickle(path_in)
    wavs = pca_wavs.transform(wavs)[:, : config.PCA_WAV_NUM_COMPONENTS]
    out_wavs = knn_wavs.predict(normalize(wavs))
    write_pickle(out_wavs, path_out)


def get_umap_steps_poses_features_embedding(dep_pickle_paths, target_pickle_paths):
    """Produce UMAP embeddings using steps and poses features.
    Fit a KNeighborsRegressor to embed new points to the UMAP.
    To emulate cosine distance, normalize the features.
    """
    scaler_stps = read_pickle(dep_pickle_paths["scaler_stp"])
    stps = np.concatenate(
        [
            read_pickle(path_stp)[read_pickle(path_idx)]
            for path_idx, path_stp in zip(
                dep_pickle_paths["idx"], dep_pickle_paths["stp"]
            )
        ]
    )
    write_pickle(stps, target_pickle_paths["stp_sample"])
    stps = scaler_stps.transform(stps)
    write_pickle(stps, target_pickle_paths["scaler_stp_sample"])
    del (
        dep_pickle_paths["idx"],
        dep_pickle_paths["stp"],
        dep_pickle_paths["scaler_stp"],
        scaler_stps,
    )
    embedding = UMAP(
        metric=config.EMB_METRIC,
        a=config.EMB_A,
        b=config.EMB_B,
        n_neighbors=config.EMB_STP_N_NEIGHBORS,
        verbose=True,
    ).fit(stps)
    write_pickle(embedding, target_pickle_paths["emb_stp"])
    write_pickle(embedding.embedding_, target_pickle_paths["out_stp"])
    knn = KNeighborsRegressor(
        n_neighbors=config.KNN_N_NEIGHBORS,
        weights=config.KNN_WEIGHTS,
        algorithm=config.KNN_ALGORITHM,
        leaf_size=config.KNN_LEAF_SIZE,
        n_jobs=config.KNN_N_JOBS,
    ).fit(normalize(stps), embedding.embedding_)
    write_pickle(knn, target_pickle_paths["knn_stp"])
    del stps, target_pickle_paths["emb_stp"], embedding


def get_umap_steps_poses_features_outcomes(scaler_stps, knn_stps, path_in, path_out):
    """UMAP outcomes from tranined KNN, using steps and poses features.
    To emulate cosine distance in KNN, normalize the features.
    """
    stps = read_pickle(path_in)
    stps = scaler_stps.transform(stps)
    out_stps = knn_stps.predict(normalize(stps))
    write_pickle(out_stps, path_out)


def get_embedding(umap_object):
    """Return UMAP coordinates"""
    if hasattr(umap_object, "embedding_"):
        return umap_object.embedding_
    elif hasattr(umap_object, "embedding"):
        return umap_object.embedding
    else:
        raise ValueError("Could not find embedding attribute of umap_object")


def get_umap_embedding_bundled_edges(path_in, path_out):
    """Save connectivity relationships of the underlying UMAP simplicial
    set data structure. Internally UMAP will make use of what can be viewed
    as a weighted graph. This graph can be plotted using the layout
    provided by UMAP as a potential diagnostic view of the embedding.
    Currently this only works for 2D embeddings.
    """
    emb = read_pickle(path_in)
    points = get_embedding(emb)
    point_df = pd.DataFrame(points, columns=("x", "y"))
    idx = point_df.index.values[:: config.EDG_SUBSAMPLE_EVERY]
    point_df = point_df.iloc[idx]
    coo_graph = emb.graph_.tocoo()
    edge_df = pd.DataFrame(
        np.vstack([coo_graph.row, coo_graph.col, coo_graph.data]).T,
        columns=("source", "target", "weight"),
    )
    edge_df["source"] = edge_df.source.astype(np.int32)
    edge_df["target"] = edge_df.target.astype(np.int32)
    edge_df = edge_df.query("source in @idx and target in @idx")
    edges = bd.hammer_bundle(point_df, edge_df, weight="weight")
    write_pickle(edges, path_out)


def _nhood_search(umap_object, nhood_size):
    if hasattr(umap_object, "_small_data") and umap_object._small_data:
        dmat = pairwise_distances(umap_object._raw_data)
        indices = np.argpartition(dmat, nhood_size)[:, :nhood_size]
        dmat_shortened = submatrix(dmat, indices, nhood_size)
        indices_sorted = np.argsort(dmat_shortened)
        indices = submatrix(indices, indices_sorted, nhood_size)
        dists = submatrix(dmat_shortened, indices_sorted, nhood_size)
    else:
        rng_state = np.empty(3, dtype=np.int64)

        indices, dists = umap_object._knn_search_index.query(
            umap_object._raw_data,
            k=nhood_size,
        )

    return indices, dists


def get_umap_embedding_local_dimension(path_in, path_out):
    """Save local dimension estimation from number of PCA components
    that explain 80% of the variance.
    """
    emb = read_pickle(path_in)
    highd_indices, _ = _nhood_search(emb, emb.n_neighbors)
    highd_indices = highd_indices[:: config.LOC_DIM_SUBSAMPLE_EVERY]
    data = emb._raw_data
    subsample_len = data[:: config.LOC_DIM_SUBSAMPLE_EVERY].shape[0]
    local_dim = np.empty(subsample_len, dtype=np.int64)
    for i in range(subsample_len):
        pca = PCA().fit(data[highd_indices[i]])
        local_dim[i] = np.where(
            np.cumsum(pca.explained_variance_ratio_) > config.LOC_DIM_VARIANCE_THRESHOLD
        )[0][0]
    write_pickle(local_dim + 1, path_out)
