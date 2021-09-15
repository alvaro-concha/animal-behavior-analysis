"""Perform HDBSCAN clustering on embedding outcomes."""
from math import ceil
from hdbscan import HDBSCAN, membership_vector, approximate_predict
from utilities import read_pickle, write_pickle
import config


def get_hdbscan_clusterers(dep_pickle_paths, target_pickle_paths):
    """Train HDBSCAN clusterers, using wavelet and angle UMAP embeddings."""
    out_angs = read_pickle(dep_pickle_paths["emb_ang"]).embedding_
    out_angs = out_angs[:: ceil(len(out_angs) / config.CLU_MAX_POINTS)]
    clu_fine_angs = HDBSCAN(
        min_cluster_size=config.CLU_FINE_MIN_SIZE,
        core_dist_n_jobs=-1,
        prediction_data=True,
    ).fit(out_angs)
    write_pickle(clu_fine_angs, target_pickle_paths["clu_fin_ang"])
    del clu_fine_angs
    clu_coarse_angs = HDBSCAN(
        min_cluster_size=config.CLU_COARSE_MIN_SIZE,
        core_dist_n_jobs=-1,
        prediction_data=True,
    ).fit(out_angs)
    write_pickle(clu_coarse_angs, target_pickle_paths["clu_coa_ang"])
    del clu_coarse_angs
    out_wavs = read_pickle(dep_pickle_paths["emb_wav"]).embedding_
    out_wavs = out_wavs[:: ceil(len(out_wavs) / config.CLU_MAX_POINTS)]
    clu_fine_wavs = HDBSCAN(
        min_cluster_size=config.CLU_FINE_MIN_SIZE,
        core_dist_n_jobs=-1,
        prediction_data=True,
    ).fit(out_wavs)
    write_pickle(clu_fine_wavs, target_pickle_paths["clu_fin_wav"])
    del clu_fine_wavs
    clu_coarse_wavs = HDBSCAN(
        min_cluster_size=config.CLU_COARSE_MIN_SIZE,
        core_dist_n_jobs=-1,
        prediction_data=True,
    ).fit(out_wavs)
    write_pickle(clu_coarse_wavs, target_pickle_paths["clu_coa_wav"])
    del clu_coarse_wavs


def get_cluster_angles_labels(
    clu_fine_angs, clu_coarse_angs, path_ang, path_fin, path_coa
):
    """Labels from trained clusterer, using angles as features."""
    out_angs = read_pickle(path_ang)
    mem_fine_angs = membership_vector(clu_fine_angs, out_angs)
    lab_fine_angs, sco_fine_angs = approximate_predict(clu_fine_angs, out_angs)
    write_pickle((mem_fine_angs, lab_fine_angs, sco_fine_angs), path_fin)
    mem_coarse_angs = membership_vector(clu_coarse_angs, out_angs)
    lab_coarse_angs, sco_coarse_angs = approximate_predict(clu_coarse_angs, out_angs)
    write_pickle((mem_coarse_angs, lab_coarse_angs, sco_coarse_angs), path_coa)


def get_cluster_wavelets_labels(
    clu_fine_wavs, clu_coarse_wavs, path_wav, path_fin, path_coa
):
    """Labels from trained clusterer, using wavelets as features."""
    out_wavs = read_pickle(path_wav)
    mem_fine_wavs = membership_vector(clu_fine_wavs, out_wavs)
    lab_fine_wavs, sco_fine_wavs = approximate_predict(clu_fine_wavs, out_wavs)
    write_pickle((mem_fine_wavs, lab_fine_wavs, sco_fine_wavs), path_fin)
    mem_coarse_wavs = membership_vector(clu_coarse_wavs, out_wavs)
    lab_coarse_wavs, sco_coarse_wavs = approximate_predict(clu_coarse_wavs, out_wavs)
    write_pickle((mem_coarse_wavs, lab_coarse_wavs, sco_coarse_wavs), path_coa)
