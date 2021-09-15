"""Animal Behavior Analysis.

Tasks
-----
    Produce UMAP embeddings using joint angles and wavelet spectra.
"""
import doit
from umap_embed import (
    get_pca_fit_wavelet_spectra,
    get_umap_embeddings,
    get_umap_angles_outcomes,
    get_umap_wavelets_outcomes,
)
from hdbscan_cluster import (
    get_hdbscan_clusterers,
    get_cluster_angles_labels,
    get_cluster_wavelets_labels,
)
import config_dodo
from utilities import read_pickle


def task_get_pca_fit_wavelet_spectra():
    """Fit incremental PCA, with whitening, to wavelet spectra features."""
    dep_pickle_paths = {"wav": []}
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths["wav"].append(config_dodo.WAV_PATH / f"wav_{pickle_end}")
    target_pickle_path = config_dodo.WAV_PATH / "pca_fit_wav.pickle"
    return {
        "file_dep": dep_pickle_paths["wav"],
        "targets": [target_pickle_path],
        "actions": [
            (
                get_pca_fit_wavelet_spectra,
                [dep_pickle_paths, target_pickle_path],
            )
        ],
    }


def task_get_umap_embeddings():
    """Produce UMAP embeddings using joint angles and wavelet spectra."""
    dep_pickle_paths = {
        "ang": [],
        "wav": [],
        "stat": config_dodo.ANG_PATH / "stat_global.pickle",
        "pca": config_dodo.WAV_PATH / "pca_fit_wav.pickle",
    }
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths["ang"].append(config_dodo.ANG_PATH / f"ang_{pickle_end}")
        dep_pickle_paths["wav"].append(config_dodo.WAV_PATH / f"wav_{pickle_end}")
    target_pickle_paths = {
        "ang_sample": config_dodo.ANG_PATH / "ang_sample.pickle",
        "wav_sample": config_dodo.WAV_PATH / "wav_sample.pickle",
        "emb_ang": config_dodo.EMB_PATH / "emb_ang_sample.pickle",
        "emb_wav": config_dodo.EMB_PATH / "emb_wav_sample.pickle",
        "out_ang": config_dodo.OUT_PATH / "out_ang_sample.pickle",
        "out_wav": config_dodo.OUT_PATH / "out_wav_sample.pickle",
    }
    return {
        "file_dep": dep_pickle_paths["ang"]
        + dep_pickle_paths["wav"]
        + [dep_pickle_paths["stat"], dep_pickle_paths["pca"]],
        "targets": list(target_pickle_paths.values()),
        "actions": [
            (
                get_umap_embeddings,
                [dep_pickle_paths, target_pickle_paths],
            )
        ],
        "verbosity": 2,
    }


def task_get_umap_angles_outcomes():
    """UMAP outcomes from trained embedding, using angles as features."""
    dep_pickle_paths = {
        "stat": config_dodo.ANG_PATH / "stat_global.pickle",
        "emb_ang": config_dodo.EMB_PATH / "emb_ang_sample.pickle",
    }
    target_pickle_paths = {}
    try:
        emb_angs = read_pickle(dep_pickle_paths["emb_ang"])
        (_, mean, var) = read_pickle(dep_pickle_paths["stat"])
        del _
    except FileNotFoundError:
        print("Oops! Might need to run get_umap_embeddings or get stat.")
        emb_angs = None
        mean, var = None, None
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths["ang"] = config_dodo.ANG_PATH / f"ang_{pickle_end}"
        target_pickle_paths["out_ang"] = config_dodo.OUT_PATH / f"out_ang_{pickle_end}"
        yield {
            "name": name,
            "file_dep": list(dep_pickle_paths.values()),
            "targets": [target_pickle_paths["out_ang"]],
            "actions": [
                (
                    get_umap_angles_outcomes,
                    [
                        emb_angs,
                        mean,
                        var,
                        dep_pickle_paths["ang"],
                        target_pickle_paths["out_ang"],
                    ],
                )
            ],
            "verbosity": 2,
        }


def task_get_umap_wavelets_outcomes():
    """UMAP outcomes from trained embedding, using wavelets as features."""
    dep_pickle_paths = {
        "pca": config_dodo.WAV_PATH / "pca_fit_wav.pickle",
        "emb_wav": config_dodo.EMB_PATH / "emb_wav_sample.pickle",
    }
    target_pickle_paths = {}
    try:
        emb_wavs = read_pickle(dep_pickle_paths["emb_wav"])
        pca = read_pickle(dep_pickle_paths["pca"])
    except FileNotFoundError:
        print("Oops! Might need to run get_umap_embeddings or get pca.")
        emb_wavs = None
        pca = None
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths["wav"] = config_dodo.WAV_PATH / f"wav_{pickle_end}"
        target_pickle_paths["out_wav"] = config_dodo.OUT_PATH / f"out_wav_{pickle_end}"
        yield {
            "name": name,
            "file_dep": list(dep_pickle_paths.values()),
            "targets": [target_pickle_paths["out_wav"]],
            "actions": [
                (
                    get_umap_wavelets_outcomes,
                    [
                        emb_wavs,
                        pca,
                        dep_pickle_paths["wav"],
                        target_pickle_paths["out_wav"],
                    ],
                )
            ],
            "verbosity": 2,
        }


def task_get_hdbscan_clusterers():
    """Train HDBSCAN clusterers, using wavelet and angle UMAP embeddings."""
    dep_pickle_paths = {
        "emb_ang": config_dodo.EMB_PATH / "emb_ang_sample.pickle",
        "emb_wav": config_dodo.EMB_PATH / "emb_wav_sample.pickle",
    }
    target_pickle_paths = {
        "clu_fin_ang": config_dodo.CLU_PATH / "clu_fin_ang.pickle",
        "clu_coa_ang": config_dodo.CLU_PATH / "clu_coa_ang.pickle",
        "clu_fin_wav": config_dodo.CLU_PATH / "clu_fin_wav.pickle",
        "clu_coa_wav": config_dodo.CLU_PATH / "clu_coa_wav.pickle",
    }
    return {
        "file_dep": list(dep_pickle_paths.values()),
        "targets": list(target_pickle_paths.values()),
        "actions": [
            (
                get_hdbscan_clusterers,
                [dep_pickle_paths, target_pickle_paths],
            )
        ],
        "verbosity": 2,
    }


def task_get_cluster_angles_labels():
    """Labels from trained clusterer, using angles as features."""
    dep_pickle_paths = {
        "clu_fin_ang": config_dodo.CLU_PATH / "clu_fin_ang.pickle",
        "clu_coa_ang": config_dodo.CLU_PATH / "clu_coa_ang.pickle",
    }
    target_pickle_paths = {}
    try:
        clu_fine_angs = read_pickle(dep_pickle_paths["clu_fin_ang"])
        clu_coarse_angs = read_pickle(dep_pickle_paths["clu_coa_ang"])
    except FileNotFoundError:
        print("Oops! Might need to run get_hdbscan_clusterers for angle labels.")
        clu_fine_angs = None
        clu_coarse_angs = None
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths["out_ang"] = config_dodo.OUT_PATH / f"out_ang_{pickle_end}"
        target_pickle_paths["lab_fin_ang"] = (
            config_dodo.LAB_PATH / f"lab_fin_ang_{pickle_end}"
        )
        target_pickle_paths["lab_coa_ang"] = (
            config_dodo.LAB_PATH / f"lab_coa_ang_{pickle_end}"
        )
        yield {
            "name": name,
            "file_dep": list(dep_pickle_paths.values()),
            "targets": list(target_pickle_paths.values()),
            "actions": [
                (
                    get_cluster_angles_labels,
                    [
                        clu_fine_angs,
                        clu_coarse_angs,
                        dep_pickle_paths["out_ang"],
                        target_pickle_paths["lab_fin_ang"],
                        target_pickle_paths["lab_coa_ang"],
                    ],
                )
            ],
            "verbosity": 2,
        }


def task_get_cluster_wavelets_labels():
    """Labels from trained clusterer, using wavelets as features."""
    dep_pickle_paths = {
        "clu_fin_wav": config_dodo.CLU_PATH / "clu_fin_wav.pickle",
        "clu_coa_wav": config_dodo.CLU_PATH / "clu_coa_wav.pickle",
    }
    target_pickle_paths = {}
    try:
        clu_fine_wavs = read_pickle(dep_pickle_paths["clu_fin_wav"])
        clu_coarse_wavs = read_pickle(dep_pickle_paths["clu_coa_wav"])
    except FileNotFoundError:
        print("Oops! Might need to run get_hdbscan_clusterers for wavelet labels.")
        clu_fine_wavs = None
        clu_coarse_wavs = None
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths["out_wav"] = config_dodo.OUT_PATH / f"out_wav_{pickle_end}"
        target_pickle_paths["lab_fin_wav"] = (
            config_dodo.LAB_PATH / f"lab_fin_wav_{pickle_end}"
        )
        target_pickle_paths["lab_coa_wav"] = (
            config_dodo.LAB_PATH / f"lab_coa_wav_{pickle_end}"
        )
        yield {
            "name": name,
            "file_dep": list(dep_pickle_paths.values()),
            "targets": list(target_pickle_paths.values()),
            "actions": [
                (
                    get_cluster_wavelets_labels,
                    [
                        clu_fine_wavs,
                        clu_coarse_wavs,
                        dep_pickle_paths["out_wav"],
                        target_pickle_paths["lab_fin_wav"],
                        target_pickle_paths["lab_coa_wav"],
                    ],
                )
            ],
            "verbosity": 2,
        }


def main():
    """Run pipeline as a Python script."""
    doit.run(globals())


if __name__ == "__main__":
    main()
