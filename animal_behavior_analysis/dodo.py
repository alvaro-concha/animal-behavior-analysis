"""Animal Behavior Analysis.

Tasks
-----
    UMAP outcomes from pre-trained embedding, using wavelets as features.
"""
from umap_embed import (
    get_umap_wavelets_embedding,
    get_umap_wavelets_outcomes,
    get_umap_steps_poses_features_embedding,
    get_umap_steps_poses_features_outcomes,
    get_umap_embedding_bundled_edges,
    get_umap_embedding_local_dimension,
)
from watershed_segment import (
    get_umap_wavelets_outcomes_densities,
    get_umap_steps_poses_features_outcomes_densities,
)
import config_dodo
from utilities import read_pickle


def task_get_umap_wavelets_embedding():
    """Produce UMAP embedding using subsampled wavelet spectra."""
    dep_pickle_paths = {
        "idx": [],
        "wav": [],
        "pca_wav": config_dodo.WAV_PATH / "pca_fit_wav.pickle",
    }
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths["idx"].append(config_dodo.IDX_PATH / f"idx_{pickle_end}")
        dep_pickle_paths["wav"].append(config_dodo.WAV_PATH / f"wav_{pickle_end}")
    target_pickle_paths = {
        "wav_sample": config_dodo.WAV_PATH / "wav_sample.pickle",
        "pca_wav_sample": config_dodo.WAV_PATH / "pca_wav_sample.pickle",
        "emb_wav": config_dodo.EMB_PATH / "emb_wav_sample.pickle",
        "out_wav": config_dodo.OUT_PATH / "out_wav_sample.pickle",
        "knn_wav": config_dodo.EMB_PATH / "knn_fit_wav.pickle",
    }
    return {
        "file_dep": dep_pickle_paths["idx"]
        + dep_pickle_paths["wav"]
        + [dep_pickle_paths["pca_wav"]],
        "targets": list(target_pickle_paths.values()),
        "actions": [
            (
                get_umap_wavelets_embedding,
                [dep_pickle_paths, target_pickle_paths],
            )
        ],
        "verbosity": 2,
    }


def task_get_umap_wavelets_outcomes():
    """UMAP outcomes from pre-trained embedding, using wavelets as features."""
    dep_pickle_paths = {
        "pca_wav": config_dodo.WAV_PATH / "pca_fit_wav.pickle",
        "knn_wav": config_dodo.EMB_PATH / "knn_fit_wav.pickle",
    }
    target_pickle_paths = {}
    try:
        pca_wavs = read_pickle(dep_pickle_paths["pca_wav"])
        knn_wavs = read_pickle(dep_pickle_paths["knn_wav"])
    except FileNotFoundError:
        print("Oops! Might need to run get_umap_wavelets_embedding or get pca.")
        pca_wavs = None
        knn_wavs = None
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
                        pca_wavs,
                        knn_wavs,
                        dep_pickle_paths["wav"],
                        target_pickle_paths["out_wav"],
                    ],
                )
            ],
            "verbosity": 2,
        }


def task_get_umap_wavelets_outcomes_densities():
    """Probability density estimations for UMAP wavelet outcomes."""
    dep_pickle_paths = {}
    target_pickle_paths = {}
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths["out_wav"] = config_dodo.OUT_PATH / f"out_wav_{pickle_end}"
        target_pickle_paths["kde_wav"] = config_dodo.WAT_PATH / f"kde_wav_{pickle_end}"
        target_pickle_paths["gau_kde_wav"] = (
            config_dodo.WAT_PATH / f"gau_kde_wav_{pickle_end}"
        )
        yield {
            "name": name,
            "file_dep": [dep_pickle_paths["out_wav"]],
            "targets": [target_pickle_paths["kde_wav"]],
            "actions": [
                (
                    get_umap_wavelets_outcomes_densities,
                    [
                        dep_pickle_paths["out_wav"],
                        target_pickle_paths["kde_wav"],
                        target_pickle_paths["gau_kde_wav"],
                    ],
                )
            ],
        }


def task_get_umap_steps_poses_features_embedding():
    """Produce UMAP embedding using subsampled steps and poses features."""
    dep_pickle_paths = {
        "idx": [],
        "stp": [],
        "scaler_stp": config_dodo.STP_PATH / "standard_scaler_fit_stp.pickle",
    }
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths["idx"].append(config_dodo.IDX_PATH / f"idx_{pickle_end}")
        dep_pickle_paths["stp"].append(config_dodo.STP_PATH / f"stp_{pickle_end}")
    target_pickle_paths = {
        "stp_sample": config_dodo.STP_PATH / "stp_sample.pickle",
        "scaler_stp_sample": config_dodo.STP_PATH / "scaler_stp_sample.pickle",
        "emb_stp": config_dodo.EMB_PATH / "emb_stp_sample.pickle",
        "out_stp": config_dodo.OUT_PATH / "out_stp_sample.pickle",
        "knn_stp": config_dodo.EMB_PATH / "knn_fit_stp.pickle",
    }
    return {
        "file_dep": dep_pickle_paths["idx"] + dep_pickle_paths["stp"],
        "targets": list(target_pickle_paths.values()),
        "actions": [
            (
                get_umap_steps_poses_features_embedding,
                [dep_pickle_paths, target_pickle_paths],
            )
        ],
        "verbosity": 2,
    }


def task_get_umap_steps_poses_features_outcomes():
    """UMAP outcomes from pre-trained embedding, steps and poses features."""
    dep_pickle_paths = {
        "knn_stp": config_dodo.EMB_PATH / "knn_fit_stp.pickle",
        "scaler_stp": config_dodo.STP_PATH / "standard_scaler_fit_stp.pickle",
    }
    target_pickle_paths = {}
    try:
        scaler_stps = read_pickle(dep_pickle_paths["scaler_stp"])
        knn_stps = read_pickle(dep_pickle_paths["knn_stp"])
    except FileNotFoundError:
        print(
            "Oops! Might need to run get_umap_steps_poses_features_embedding or get scaler."
        )
        scaler_stps = None
        knn_stps = None
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths["stp"] = config_dodo.STP_PATH / f"stp_{pickle_end}"
        target_pickle_paths["out_stp"] = config_dodo.OUT_PATH / f"out_stp_{pickle_end}"
        yield {
            "name": name,
            "file_dep": list(dep_pickle_paths.values()),
            "targets": [target_pickle_paths["out_stp"]],
            "actions": [
                (
                    get_umap_steps_poses_features_outcomes,
                    [
                        scaler_stps,
                        knn_stps,
                        dep_pickle_paths["stp"],
                        target_pickle_paths["out_stp"],
                    ],
                )
            ],
            "verbosity": 2,
        }


def task_get_umap_steps_poses_features_outcomes_densities():
    """Probability density estimations for UMAP steps and poses outcomes."""
    dep_pickle_paths = {}
    target_pickle_paths = {}
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths["out_stp"] = config_dodo.OUT_PATH / f"out_stp_{pickle_end}"
        target_pickle_paths["kde_stp"] = config_dodo.WAT_PATH / f"kde_stp_{pickle_end}"
        target_pickle_paths["gau_kde_stp"] = (
            config_dodo.WAT_PATH / f"gau_kde_stp_{pickle_end}"
        )
        yield {
            "name": name,
            "file_dep": [dep_pickle_paths["out_stp"]],
            "targets": [target_pickle_paths["kde_stp"]],
            "actions": [
                (
                    get_umap_steps_poses_features_outcomes_densities,
                    [
                        dep_pickle_paths["out_stp"],
                        target_pickle_paths["kde_stp"],
                        target_pickle_paths["gau_kde_stp"],
                    ],
                )
            ],
        }


def task_get_umap_wavelets_bundled_edges():
    """Bundled edges for UMAP wavelet outcomes."""
    dep_pickle_path = config_dodo.EMB_PATH / "emb_wav_sample.pickle"
    target_pickle_path = config_dodo.EMB_PATH / "edg_wav_sample.pickle"
    return {
        "file_dep": [dep_pickle_path],
        "targets": [target_pickle_path],
        "actions": [
            (
                get_umap_embedding_bundled_edges,
                [
                    dep_pickle_path,
                    target_pickle_path,
                ],
            )
        ],
    }


def task_get_umap_steps_poses_features_bundled_edges():
    """Bundled edges for UMAP steps and poses outcomes."""
    dep_pickle_path = config_dodo.EMB_PATH / "emb_stp_sample.pickle"
    target_pickle_path = config_dodo.EMB_PATH / "edg_stp_sample.pickle"
    return {
        "file_dep": [dep_pickle_path],
        "targets": [target_pickle_path],
        "actions": [
            (
                get_umap_embedding_bundled_edges,
                [
                    dep_pickle_path,
                    target_pickle_path,
                ],
            )
        ],
    }


def task_get_umap_wavelets_local_dimension():
    """Local dimension for UMAP wavelet outcomes."""
    dep_pickle_path = config_dodo.EMB_PATH / "emb_wav_sample.pickle"
    target_pickle_path = config_dodo.EMB_PATH / "loc_dim_wav_sample.pickle"
    return {
        "file_dep": [dep_pickle_path],
        "targets": [target_pickle_path],
        "actions": [
            (
                get_umap_embedding_local_dimension,
                [
                    dep_pickle_path,
                    target_pickle_path,
                ],
            )
        ],
    }


def task_get_umap_steps_poses_features_local_dimension():
    """Local dimension for UMAP steps and poses outcomes."""
    dep_pickle_path = config_dodo.EMB_PATH / "emb_stp_sample.pickle"
    target_pickle_path = config_dodo.EMB_PATH / "loc_dim_stp_sample.pickle"
    return {
        "file_dep": [dep_pickle_path],
        "targets": [target_pickle_path],
        "actions": [
            (
                get_umap_embedding_local_dimension,
                [
                    dep_pickle_path,
                    target_pickle_path,
                ],
            )
        ],
    }


# I need some tasks to: save all_wavs (UMAP outcomes) and save all_kde (all wav kde)
# THESE ARE IN DATA_AGGREGATE, FOR NOW

# out_wav_all:
# all_wavs = []
# for key in config_dodo.KEY_LIST:
#     name = config_dodo.SUBJECT_NAME.format(*key)
#     pickle_end = name + ".pickle"
#     path_out_wav = config_dodo.OUT_PATH / f"out_wav_{pickle_end}"
#     out_wavs = read_pickle(path_out_wav)
#     all_wavs.append(out_wavs)
# all_wavs = np.concatenate(all_wavs)
# write_pickle(all_wavs, config_dodo.OUT_PATH / f"out_wav_all.pickle")

# kde_wav_all:
# from fastkde import fastKDE
# from skimage.util import img_as_ubyte
# def dynamic_range(img):
#     """
#     Returns image with full dynamic range
#     Parameters
#     ----------
#     img: array_like
#             Image whose intensities are to be rescaled to full dynamic range
#     """
#     img = (img - img.min()) / (img.max() - img.min())
#     img = img_as_ubyte(img)
#     return img
# all_kde, _ = fastKDE.pdf(
#     all_wavs[:, 0],
#     all_wavs[:, 1],
#     ...,
# )
# all_kde = dynamic_range(all_kde)
# write_pickle(all_kde, config_dodo.WAT_PATH / f"kde_wav_all.pickle")


def main():
    """Run pipeline as a Python script."""
    import doit

    doit.run(globals())


if __name__ == "__main__":
    main()
