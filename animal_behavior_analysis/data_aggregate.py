"""Aggregate data.

Compute and save entropy for each trial, and total
Compute and save PDF estimations for each trial, and total
Use fastKDE with axes argument
And then apply gaussian convolution, with same sigma as used in watershed
Update watershed segmentation calculation
Redo watershed

entropies = pd.DataFrame({"mouse":m, "day":d, "trial":t, "ent":ent})
Puedo agregar una entrada {"mouse":"all", "day":"all", "trial":"all", "ent":ent}

Compute and save geometric mean frequency and argmax, argmin frequency"""
import multiprocessing as mp
from copy import deepcopy
from collections import defaultdict
import numpy as np
import config_dodo
import config
from utilities import read_pickle, write_pickle
from skimage.util import img_as_ubyte
from watershed_segment import (
    get_umap_wavelets_outcomes_densities,
    get_wavelets_watershed_segmentation,
    get_umap_steps_poses_features_outcomes_densities,
    get_steps_poses_features_watershed_segmentation,
)
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import entropy

(config_dodo.FIG_PATH / "Watershed").mkdir(parents=True, exist_ok=True)
sns.set_context("paper", font_scale=2.0)
sns.set_style(
    "ticks",
    {
        "text.color": "k",
        "axes.edgecolor": "k",
        "axes.labelcolor": "k",
        "xtick.color": "k",
        "ytick.color": "k",
    },
)


def get_wav_out_all():
    dep_pickle_paths = {"out_wav": []}
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        print(name)
        pickle_end = name + ".pickle"
        dep_pickle_paths["out_wav"].append(
            config_dodo.OUT_PATH / f"out_wav_{pickle_end}"
        )
    out_wavs = np.concatenate(
        [read_pickle(path) for path in dep_pickle_paths["out_wav"]]
    )
    write_pickle(out_wavs, config_dodo.OUT_PATH / "out_wav_all.pickle")


def get_dynamic_range(img):
    """
    Returns image with full dynamic range.

    Parameters
    ----------
    img: array_like
        Image whose intensities are to be rescaled to full dynamic range
    """
    img = (img - img.min()) / (img.max() - img.min())
    img = img_as_ubyte(img)
    return img


def get_wav_kde_all():
    path_out_wavs = config_dodo.OUT_PATH / "out_wav_all.pickle"
    path_kde = config_dodo.WAT_PATH / "kde_wav_all.pickle"
    path_gau_kde = config_dodo.WAT_PATH / "gau_kde_wav_all.pickle"
    get_umap_wavelets_outcomes_densities(path_out_wavs, path_kde, path_gau_kde)


def get_wav_lab_all_watershed_segment():
    out_wavs = read_pickle(config_dodo.OUT_PATH / "out_wav_all.pickle")
    kde_wavs = read_pickle(config_dodo.WAT_PATH / "kde_wav_all.pickle")
    path_labels = config_dodo.LAB_PATH / "lab_wav_all.pickle"
    path_segmentation = config_dodo.WAT_PATH / "segmentation_wav_all.pickle"
    get_wavelets_watershed_segmentation(
        out_wavs=out_wavs,
        img=kde_wavs,
        path_labels=path_labels,
        path_segmentation=path_segmentation,
    )


def annotate_labels(ax, unique_labels, labels, outcomes):
    """
    Annotates labels to a UMAP segmented embedding
    Parameters
    ----------
    ax: plt.axes
            Axes to annotate labels on
    unique_labels: array_like
            Unique label values
    labels: array_like
            Labels associated to the UMAP outcomes
    outcomes: array_like
            UMAP outcomes, the annotations will be displayed in their centroids
    """
    for i in unique_labels:
        label_median = np.median(outcomes[labels == i], axis=0)
        label_txt = ax.text(
            label_median[0],
            label_median[1],
            str(i),
            fontsize=30,
            # color="k",
            color="1",
            horizontalalignment="center",
            verticalalignment="center",
        )
        label_txt.set_path_effects(
            [
                PathEffects.Stroke(
                    linewidth=3,
                    # foreground="w",
                    foreground="0",
                ),
                PathEffects.Normal(),
            ]
        )


def plot_wav_watershed_segmentation_white_background():
    out_wavs = read_pickle(config_dodo.OUT_PATH / "out_wav_all.pickle")
    labels = read_pickle(config_dodo.LAB_PATH / "lab_wav_all.pickle")
    unique_labels = np.unique(labels)
    segmentation = read_pickle(config_dodo.WAT_PATH / "segmentation_wav_all.pickle")
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    plt.xlabel(r"UMAP$_1$")
    plt.ylabel(r"UMAP$_2$")
    plt.imshow(
        segmentation,
        origin="lower",
        cmap="binary",
        alpha=0.5,
        extent=config.WAT_EMB_WAV_LIM * 2,
    )
    plt.scatter(
        *out_wavs[::10].T,
        s=1,
        alpha=0.0025,
        c=labels[::10],
        cmap="tab10",
    )
    annotate_labels(ax, unique_labels, labels, out_wavs)
    sns.despine(trim=True)
    plt.axis("equal")
    plt.savefig(
        config_dodo.FIG_PATH / "Watershed/segmentacion_watershed_wav.png",
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()


def plot_wav_ethogram_segment():
    out_wavs = read_pickle(config_dodo.OUT_PATH / "out_wav_all.pickle")
    labels = read_pickle(config_dodo.LAB_PATH / "lab_wav_all.pickle")
    #
    plt.figure(figsize=(7, 7))
    plt.scatter(
        *out_wavs[::10].T,
        s=1,
        alpha=0.0025,
        c=labels[::10],
        cmap="tab10",
    )
    plt.xlabel(r"UMAP$_1$")
    plt.ylabel(r"UMAP$_2$")
    plt.xlim(*config.WAT_EMB_WAV_LIM)
    plt.ylim(*config.WAT_EMB_WAV_LIM)
    annotate_labels(plt.gca(), np.unique(labels), labels, out_wavs)
    sns.despine(trim=True)
    plt.axis("equal")
    plt.savefig(
        config_dodo.FIG_PATH / "Watershed/etograma_anotado_wav.png",
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()


# Here come Steps and Poses


def get_stp_out_all():
    dep_pickle_paths = {"out_stp": []}
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        print(name)
        pickle_end = name + ".pickle"
        dep_pickle_paths["out_stp"].append(
            config_dodo.OUT_PATH / f"out_stp_{pickle_end}"
        )
    out_stps = np.concatenate(
        [read_pickle(path) for path in dep_pickle_paths["out_stp"]]
    )
    write_pickle(out_stps, config_dodo.OUT_PATH / "out_stp_all.pickle")


def get_stp_kde_all():
    path_out_stps = config_dodo.OUT_PATH / "out_stp_all.pickle"
    path_kde = config_dodo.WAT_PATH / "kde_stp_all.pickle"
    path_gau_kde = config_dodo.WAT_PATH / "gau_kde_stp_all.pickle"
    get_umap_steps_poses_features_outcomes_densities(
        path_out_stps, path_kde, path_gau_kde
    )


def get_stp_lab_all_watershed_segment():
    out_stps = read_pickle(config_dodo.OUT_PATH / "out_stp_all.pickle")
    kde_stps = read_pickle(config_dodo.WAT_PATH / "kde_stp_all.pickle")
    path_labels = config_dodo.LAB_PATH / "lab_stp_all.pickle"
    path_segmentation = config_dodo.WAT_PATH / "segmentation_stp_all.pickle"
    get_steps_poses_features_watershed_segmentation(
        out_stps=out_stps,
        img=kde_stps,
        path_labels=path_labels,
        path_segmentation=path_segmentation,
    )


def plot_stp_watershed_segmentation_white_background():
    out_stps = read_pickle(config_dodo.OUT_PATH / "out_stp_all.pickle")
    labels = read_pickle(config_dodo.LAB_PATH / "lab_stp_all.pickle")
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 10:
        cmap = "tab10"
    else:
        cmap = "tab20"
    segmentation = read_pickle(config_dodo.WAT_PATH / "segmentation_stp_all.pickle")
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    plt.xlabel(r"UMAP$_1$")
    plt.ylabel(r"UMAP$_2$")
    plt.imshow(
        segmentation,
        origin="lower",
        cmap="binary",
        alpha=0.5,
        extent=config.WAT_EMB_STP_LIM * 2,
    )
    plt.scatter(
        *out_stps[::10].T,
        s=1,
        alpha=0.0025,
        c=labels[::10],
        cmap=cmap,
    )
    annotate_labels(ax, unique_labels, labels, out_stps)
    sns.despine(trim=True)
    plt.axis("equal")
    plt.savefig(
        config_dodo.FIG_PATH / "Watershed/segmentacion_watershed_stp.png",
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()


def plot_stp_ethogram_segment():
    out_stps = read_pickle(config_dodo.OUT_PATH / "out_stp_all.pickle")
    labels = read_pickle(config_dodo.LAB_PATH / "lab_stp_all.pickle")
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 10:
        cmap = "tab10"
    else:
        cmap = "tab20"
    plt.figure(figsize=(7, 7))
    plt.scatter(
        *out_stps[::10].T,
        s=1,
        alpha=0.0025,
        c=labels[::10],
        cmap=cmap,
    )
    plt.xlabel(r"UMAP$_1$")
    plt.ylabel(r"UMAP$_2$")
    plt.xlim(*config.WAT_EMB_STP_LIM)
    plt.ylim(*config.WAT_EMB_STP_LIM)
    annotate_labels(plt.gca(), np.unique(labels), labels, out_stps)
    sns.despine(trim=True)
    plt.axis("equal")
    plt.savefig(
        config_dodo.FIG_PATH / "Watershed/etograma_anotado_stp.png",
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()


def get_plot_latencies():
    """Get latencies."""
    frame_rate = 100.0
    latencies = {}
    num_frames = {}
    lat_fra_ratios = {}
    for key in config_dodo.KEY_LIST:
        # plt.figure(figsize=(10, 4))
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        path_kal = config_dodo.KAL_PATH / f"kal_xy_{pickle_end}"
        variables = read_pickle(path_kal)[:, (0, 1, 2), 1]
        variables[variables < -100.0] = -100.0
        variables[variables > 150.0] = 150.0
        # plt.plot(variables, alpha=0.5, lw=1)
        volatility = pd.DataFrame(variables).rolling(250).std().mean(axis=1) * 10
        volatility = np.gradient(savgol_filter(volatility, 501, 3)) * 500
        volatility = volatility / np.nanmax(volatility) * 100.0
        # plt.plot(volatility, alpha=0.9)
        peaks, properties = find_peaks(volatility, height=20.0, prominence=20.0)
        latencies[key] = peaks[-1] / frame_rate
        num_frames[key] = len(variables)
        lat_fra_ratios[key] = latencies[key] / num_frames[key]
        # plt.plot(peaks, volatility[peaks], "x", c="k")
        # plt.vlines(
        #     x=peaks,
        #     ymin=volatility[peaks] - properties["prominences"],
        #     ymax=volatility[peaks],
        #     alpha=0.9,
        #     color="k",
        # )
        # plt.ylim(-100, 150)
        # plt.title(name)
        # sns.despine(trim=True)
        # plt.show()
        # plt.close()
    key = (282, 5, 1)  # Only exception to the rule
    latencies[key] = 19004 / frame_rate
    lat_fra_ratios[key] = 19004 / num_frames[key]

    data = []
    for key in latencies.keys():
        data.append(
            {
                "mouse": key[0],
                "day": key[1],
                "trial": key[2],
                "num_frames": num_frames[key],
                "lat": latencies[key],
            }
        )
    data = pd.DataFrame(data)

    plt.figure(figsize=(7, 4))
    # sns.pointplot(x="day", y="lat", data=data, capsize=0.1, join=True, ci=68)
    sns.pointplot(
        x="day",
        y="lat",
        hue="mouse",
        dodge=0.5,
        data=data,
        capsize=0.1,
        join=True,
        ci=68,
    )
    # sns.pointplot(x="day", y="lat", c="k", data=data, capsize=0.1, join=True, ci=68)
    # sns.lineplot(x="day", y="lat", palette=["k"], data=data, ci=68)
    sns.pointplot(
        x="day",
        y="lat",
        color="k",
        # ls="dashed",
        linestyle="dashed",
        # palette=["k"],
        # ls=["--"],
        data=data,
        capsize=0.1,
        join=True,
        ci=68,
    )
    # sns.swarmplot(x="day", y="lat", data=data, color="k", alpha=0.5)
    plt.ylim(0, 310)
    plt.gca().set_yticks(range(0, 400, 100))
    sns.despine(trim=True)
    plt.show()
    plt.figure(figsize=(7, 4))
    sns.pointplot(
        x="day",
        y="lat",
        hue="day",
        data=data,
        capsize=0.1,
        dodge=0.5,
        join=False,
        ci=68,  # ci=68 is an estimate of the SEM (Standard Error of the Mean), if data is Gaussian
    )
    plt.ylim(0, 300)
    plt.gca().set_yticks(range(0, 400, 100))
    sns.despine(trim=True)
    plt.show()

    plt.figure(figsize=(7, 4))
    sns.pointplot(
        x="day",
        y="lat",
        hue="trial",
        data=data,
        capsize=0.1,
        dodge=0.5,
        join=False,
        ci=68,  # ci=68 is an estimate of the SEM (Standard Error of the Mean), if data is Gaussian
    )
    plt.ylim(0, 300)
    plt.gca().set_yticks(range(0, 400, 100))
    sns.despine(trim=True)
    plt.show()


def get_single_entropy_gau_kde(key):
    name = config_dodo.SUBJECT_NAME.format(*key)
    print(name)
    pickle_end = name + ".pickle"
    path_gau_kde_trial = config_dodo.WAT_PATH / f"gau_kde_wav_{pickle_end}"
    gau_kde_trial = read_pickle(path_gau_kde_trial)
    return key, entropy(gau_kde_trial.ravel())


def get_single_entropy_kde(key):
    name = config_dodo.SUBJECT_NAME.format(*key)
    print(name)
    pickle_end = name + ".pickle"
    path_kde_trial = config_dodo.WAT_PATH / f"kde_wav_{pickle_end}"
    kde_trial = read_pickle(path_kde_trial)
    return key, entropy(kde_trial.ravel())


def get_plot_entropies():
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(get_single_entropy_gau_kde, config_dodo.KEY_LIST)
        pool.close()
        pool.join()
    entropies = dict(results)
    gau_kde_all = read_pickle(config_dodo.WAT_PATH / "gau_kde_wav_all.pickle")
    total_entropy = entropy(gau_kde_all.ravel())
    data = []
    for key in entropies.keys():
        data.append(
            {
                "mouse": key[0],
                "day": key[1],
                "trial": key[2],
                "ent": entropies[key] / total_entropy,
            }
        )
    data = pd.DataFrame(data)

    plt.figure(figsize=(7, 4))
    sns.pointplot(x="day", y="ent", data=data, capsize=0.1, join=True, ci=68)
    sns.swarmplot(x="day", y="ent", data=data, color="k", alpha=0.5)
    sns.despine(trim=True)
    plt.show()

    plt.figure(figsize=(7, 4))
    sns.pointplot(
        x="day",
        y="ent",
        hue="day",
        data=data,
        capsize=0.1,
        dodge=0.5,
        join=False,
        ci=68,  # ci=68 is an estimate of the SEM (Standard Error of the Mean), if data is Gaussian
    )
    sns.despine(trim=True)
    plt.show()

    plt.figure(figsize=(7, 4))
    sns.pointplot(
        x="day",
        y="ent",
        hue="trial",
        data=data,
        capsize=0.1,
        dodge=0.5,
        join=False,
        ci=68,  # ci=68 is an estimate of the SEM (Standard Error of the Mean), if data is Gaussian
    )
    sns.despine(trim=True)
    plt.show()


def get_single_entropy_labels(key, labels, data, hist_bins):
    print(key)
    m, d, t = key
    idx = data.query(r"mouse==@m & day==@d & trial==@t")["global_frame_idx"]
    trial_labels = labels[idx]
    hist_trial_labels = np.histogram(a=trial_labels, bins=hist_bins)[0] / len(
        trial_labels
    )
    entropy_labels = entropy(hist_trial_labels)
    return key, (entropy_labels, hist_trial_labels)


def get_plot_trial_label_entropy_histograms():
    data = {
        "mouse": [],
        "day": [],
        "trial": [],
        "global_frame_idx": [],
        "local_frame_idx": [],
    }
    len_cum = 0
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        path_out_wav = config_dodo.OUT_PATH / f"out_wav_{pickle_end}"
        out_wavs = read_pickle(path_out_wav)
        len_trial = len(out_wavs)
        data["mouse"].append(np.full(len_trial, key[0]))
        data["day"].append(np.full(len_trial, key[1]))
        data["trial"].append(np.full(len_trial, key[2]))
        data["local_frame_idx"].append(np.arange(len_trial))
        data["global_frame_idx"].append(np.arange(len_cum, len_cum + len_trial))
        len_cum += len_trial
        # plt.figure(figsize=(10, 4))
        # plt.plot(out_wavs, alpha=0.9)
        # plt.title(name)
        # plt.show()
        # plt.close()
    for key, value in data.items():
        data[key] = np.concatenate(value)
    data = pd.DataFrame(data)

    labels = read_pickle(config_dodo.LAB_PATH / "lab_wav_all.pickle")
    num_labels = labels.max()
    hist_bins = np.arange(0.5, num_labels + 1.5)
    hist_labels = np.histogram(a=labels, bins=hist_bins)[0] / len(labels)
    total_entropy_lab = entropy(hist_labels)
    #
    # plt.figure(figsize=(10, 4))
    # sns.histplot(
    #     x=labels,
    #     bins=np.arange(0.5, num_labels + 1.5),
    #     hue=labels,
    #     palette="tab10",
    #     shrink=0.8,
    #     legend=False,
    # )
    # ax = plt.gca()
    # ax.set_xticks(range(1, num_labels + 1))
    # plt.title("All labels")
    # sns.despine(trim=True)
    # plt.show()

    # for key in config_dodo.KEY_LIST:
    #
    #     plt.figure(figsize=(10, 4))
    #     sns.histplot(
    #         x=trial_labels,
    #         bins=np.arange(0.5, num_labels + 1.5),
    #         hue=trial_labels,
    #         palette="tab10",
    #         shrink=0.8,
    #         legend=False,
    #     )
    #     ax = plt.gca()
    #     ax.set_xticks(range(1, num_labels + 1))
    #     plt.title(f"{key}")
    #     sns.despine(trim=True)
    #     plt.show()

    #     events = [
    #         [i for i, l in enumerate(trial_labels) if l == label]
    #         for label in range(1, num_labels + 1)
    #     ]
    #     colors = cm.get_cmap("tab10", num_labels)(range(num_labels))
    #     plt.figure(figsize=(10, 4))
    #     plt.eventplot(events, colors=colors)
    #     plt.title(f"{key}")
    #     sns.despine(trim=True)
    #     plt.show()

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(
            get_single_entropy_labels,
            [(key, labels, data, hist_bins) for key in config_dodo.KEY_LIST],
        )
        pool.close()
        pool.join()
    entropies_lab = dict(results)
    data_ent_lab = []
    for key, (ent, hist) in entropies_lab.items():
        data_ent_lab.append(
            {
                "mouse": key[0],
                "day": key[1],
                "trial": key[2],
                "ent": ent / total_entropy_lab,
                "hist": hist,
            }
        )
    data_ent_lab = pd.DataFrame(data_ent_lab)

    plt.figure(figsize=(7, 4))
    sns.pointplot(x="day", y="ent", data=data_ent_lab, capsize=0.1, join=True, ci=68)
    sns.swarmplot(x="day", y="ent", data=data_ent_lab, color="k", alpha=0.5)
    sns.despine(trim=True)
    plt.show()

    plt.figure(figsize=(7, 4))
    sns.pointplot(
        x="day",
        y="ent",
        hue="day",
        data=data_ent_lab,
        capsize=0.1,
        dodge=0.5,
        join=False,
        ci=68,  # ci=68 is an estimate of the SEM (Standard Error of the Mean), if data is Gaussian
    )
    sns.despine(trim=True)
    plt.show()

    plt.figure(figsize=(7, 4))
    sns.pointplot(
        x="day",
        y="ent",
        hue="trial",
        data=data_ent_lab,
        capsize=0.1,
        dodge=0.5,
        join=False,
        ci=68,  # ci=68 is an estimate of the SEM (Standard Error of the Mean), if data is Gaussian
    )
    sns.despine(trim=True)
    plt.show()


def get_geometric_mean_frequencies(wavs):
    freq_geoms = []
    for i in range(len(config.ANG_MARKER_IDX)):
        weights = (
            wavs[:, config.WAV_NUM_CHANNELS * i : config.WAV_NUM_CHANNELS * (i + 1)]
            ** 2
        )
        freq_geom = 10 ** (
            np.sum(np.log10(config.WAV_F_CHANNELS[np.newaxis, :]) * weights, axis=1)
            / np.sum(weights, axis=1)
        )
        freq_geoms.append(freq_geom)
    return np.array(freq_geoms)


def get_angle_wavelet_long_data(long_data):
    """Get angle and wavelet long data."""
    couple_angles_dict = {
        "ang_hind_left": 13,
        "ang_hind_right": 17,
        "ang_quad_left": 14,
        "ang_quad_right": 16,
        "ang_front_left": 2,
        "ang_front_right": 1,
        "ang_head": 11,
        "ang_tail": 21,
    }
    freq_angles_dict = {
        "freq_hind": np.arange(13, 19),
        "freq_tail": np.arange(19, 28),
        "freq_front": np.arange(3),
        "freq_head": np.arange(3, 13),
        "freq_all": np.arange(28),
    }
    pca = read_pickle(config_dodo.WAV_PATH / "pca_fit_wav.pickle")
    pca = pca.set_params(n_components=config.PCA_WAV_NUM_COMPONENTS)
    pca.components_ = pca.components_[: config.PCA_WAV_NUM_COMPONENTS]
    pca.explained_variance_ = pca.explained_variance_[: config.PCA_WAV_NUM_COMPONENTS]
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        print(name)
        pickle_end = name + ".pickle"
        path_ang = config_dodo.ANG_PATH / f"ang_{pickle_end}"
        angs = read_pickle(path_ang)
        for ang_key, idx in couple_angles_dict.items():
            long_data[ang_key].append(
                savgol_filter(
                    angs[:, idx],
                    window_length=21,
                    polyorder=2,
                    axis=0,
                )
            )
        path_wav = config_dodo.WAV_PATH / f"wav_{pickle_end}"
        wavs = read_pickle(path_wav)
        wavs = pca.transform(wavs)[:, : config.PCA_WAV_NUM_COMPONENTS]
        wavs = pca.inverse_transform(wavs)
        freq_geoms = get_geometric_mean_frequencies(wavs)
        long_data["argmin_freq"].append(np.argmin(freq_geoms, axis=0))
        long_data["argmax_freq"].append(np.argmax(freq_geoms, axis=0))
        for freq_key, idx in freq_angles_dict.items():
            long_data[freq_key].append(np.mean(freq_geoms[idx], axis=0))
    return long_data


def get_long_wav_data():
    """Get long data, detailed for each frame of video."""
    long_data = defaultdict(list)
    len_cum = 0
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        path_out_wav = config_dodo.OUT_PATH / f"out_wav_{pickle_end}"
        out_wavs = read_pickle(path_out_wav)
        len_trial = len(out_wavs)
        long_data["mouse"].append(np.full(len_trial, key[0]))
        long_data["day"].append(np.full(len_trial, key[1]))
        long_data["trial"].append(np.full(len_trial, key[2]))
        long_data["local_frame_idx"].append(np.arange(len_trial))
        long_data["global_frame_idx"].append(np.arange(len_cum, len_cum + len_trial))
        len_cum += len_trial
    print("Now angles wavelets")
    long_data = get_angle_wavelet_long_data(long_data)

    for key, value in long_data.items():
        long_data[key] = np.concatenate(value)
        print(key, len(long_data[key]))
    long_data = pd.DataFrame(long_data)
    labels = read_pickle(config_dodo.LAB_PATH / "lab_wav_all.pickle")
    long_data["label"] = labels
    long_data["sequence_group"] = (
        (
            long_data[["mouse", "day", "trial", "label"]]
            != long_data[["mouse", "day", "trial", "label"]].shift()
        )
        .any(axis=1)
        .cumsum()
    )

    print(long_data)
    print(long_data.keys())
    write_pickle(long_data, config_dodo.AGG_PATH / "long_data.pickle")


def get_phase_shift_angles(masked_long_data, key_tuple):
    x1 = masked_long_data[key_tuple[0]]
    x2 = masked_long_data[key_tuple[1]]
    x1 = (x1 - x1.mean()) / x1.std()
    x2 = (x2 - x2.mean()) / x2.std()
    cos_sim = (x1 @ x2.T) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    return np.arccos(cos_sim) / np.pi


def get_sequence_wav_data():
    """Get label sequence data, condensing consecutive repeating labels."""
    long_data = read_pickle(config_dodo.AGG_PATH / "long_data.pickle")
    freq_angles_list = [
        "freq_hind",
        "freq_tail",
        "freq_front",
        "freq_head",
        "freq_all",
    ]
    long_data_group = long_data.groupby(["sequence_group"])
    seq_data = pd.concat(
        [
            long_data_group[["mouse", "day", "trial", "label", "sequence_group"]]
            .median()
            .applymap(int),
            long_data_group["sequence_group"].count().rename("duration"),
            long_data_group["sequence_group"]
            .count()
            .apply(np.log10)
            .rename("log_duration"),
            long_data_group[["local_frame_idx", "global_frame_idx"]].first(),
            long_data_group[freq_angles_list].mean(),
            long_data_group[["argmin_freq", "argmax_freq"]].median().applymap(int),
        ],
        axis=1,
    )
    short_couple_angles_dict = {
        "phase_hind": ("ang_hind_left", "ang_hind_right"),
        "phase_quad": ("ang_quad_left", "ang_quad_right"),
        "phase_front": ("ang_front_left", "ang_front_right"),
        "phase_head_tail": ("ang_head", "ang_tail"),
    }
    phase_list = [
        long_data_group.apply(get_phase_shift_angles, key_tuple=key_tuple).rename(
            couple
        )
        for couple, key_tuple in short_couple_angles_dict.items()
    ]
    seq_data = pd.concat([seq_data] + phase_list, axis=1)
    seq_data["str_label"] = seq_data["label"].apply(str)
    seq_data = seq_data.reset_index(drop=True)

    print(seq_data)
    print(seq_data.keys())
    write_pickle(seq_data, config_dodo.AGG_PATH / "sequence_data.pickle")


def get_short_wav_data():
    """Get short data, about entire trials."""
    """Get latencies."""
    frame_rate = 100.0
    latencies = {}
    num_frames = {}
    lat_fra_ratios = {}
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        path_kal = config_dodo.KAL_PATH / f"kal_xy_{pickle_end}"
        variables = read_pickle(path_kal)[:, (0, 1, 2), 1]
        variables[variables < -100.0] = -100.0
        variables[variables > 150.0] = 150.0
        volatility = pd.DataFrame(variables).rolling(250).std().mean(axis=1) * 10
        volatility = np.gradient(savgol_filter(volatility, 501, 3)) * 500
        volatility = volatility / np.nanmax(volatility) * 100.0
        peaks, _ = find_peaks(volatility, height=20.0, prominence=20.0)
        latencies[key] = peaks[-1] / frame_rate
        num_frames[key] = len(variables)
        lat_fra_ratios[key] = latencies[key] / num_frames[key]

    key = (282, 5, 1)  # Only exception to the rule
    latencies[key] = 19004 / frame_rate
    lat_fra_ratios[key] = 19004 / num_frames[key]

    """Get entropies"""
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(get_single_entropy_gau_kde, config_dodo.KEY_LIST)
        pool.close()
        pool.join()
    entropies_gau_kde = dict(results)
    gau_kde_all = read_pickle(config_dodo.WAT_PATH / "gau_kde_wav_all.pickle")
    total_entropy_gau_kde = entropy(gau_kde_all.ravel())

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(get_single_entropy_kde, config_dodo.KEY_LIST)
        pool.close()
        pool.join()
    entropies_kde = dict(results)
    kde_all = read_pickle(config_dodo.WAT_PATH / "kde_wav_all.pickle")
    total_entropy_kde = entropy(kde_all.ravel())

    """Get label entropy and histograms."""
    labels = read_pickle(config_dodo.LAB_PATH / "lab_wav_all.pickle")
    long_data = read_pickle(config_dodo.AGG_PATH / "long_data.pickle")
    num_labels = labels.max()
    hist_bins = np.arange(0.5, num_labels + 1.5)
    hist_labels = np.histogram(a=labels, bins=hist_bins)[0] / len(labels)
    total_entropy_lab = entropy(hist_labels)

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(
            get_single_entropy_labels,
            [(key, labels, long_data, hist_bins) for key in config_dodo.KEY_LIST],
        )
        pool.close()
        pool.join()
    entropies_lab = dict(results)

    """Get angles, wavelet frequencies."""
    short_couple_angles_dict = {
        "phase_hind": ("ang_hind_left", "ang_hind_right"),
        "phase_quad": ("ang_quad_left", "ang_quad_right"),
        "phase_front": ("ang_front_left", "ang_front_right"),
        "phase_head_tail": ("ang_head", "ang_tail"),
    }
    freq_angles_list = [
        "freq_hind",
        "freq_tail",
        "freq_front",
        "freq_head",
        "freq_all",
    ]

    short_data = []
    for key, (ent_lab, hist) in entropies_lab.items():
        short_data.append(
            {
                "mouse": key[0],
                "day": key[1],
                "trial": key[2],
                "ent_lab": ent_lab / total_entropy_lab,
                "ent_kde": entropies_kde[key] / total_entropy_kde,
                "ent_gau_kde": entropies_gau_kde[key] / total_entropy_gau_kde,
                "lat": latencies[key],
                "num_frames": num_frames[key],
                "lat_fra_ratio": lat_fra_ratios[key],
                "high_performance": key[0] in [265, 297, 329],
                "tanda_1": key[0] in [262, 263, 264, 265, 282],
                "tanda_2": key[0] in [295, 297, 298, 329, 330],
            }
        )
        m, d, t = key
        masked_long_data = long_data.query(r"mouse==@m & day==@d & trial==@t")
        short_data[-1]["argmin_freq"] = int(
            np.nanmedian(masked_long_data["argmin_freq"])
        )
        short_data[-1]["argmax_freq"] = int(
            np.nanmedian(masked_long_data["argmax_freq"])
        )
        for idx in range(num_labels):
            l = idx + 1
            masked_long_data = long_data.query(
                r"mouse==@m & day==@d & trial==@t & label==@l"
            )
            short_data[-1][f"prob_l{l}"] = hist[idx]
            for freq_key in freq_angles_list:
                short_data[-1][f"{freq_key}_l{l}"] = masked_long_data[freq_key].mean()
            for couple, key_tuple in short_couple_angles_dict.items():
                short_data[-1][f"{couple}_l{l}"] = get_phase_shift_angles(
                    masked_long_data, key_tuple
                )
    short_data = pd.DataFrame(short_data)

    print(short_data)
    print(short_data.keys())
    write_pickle(short_data, config_dodo.AGG_PATH / "short_data.pickle")


if __name__ == "__main__":

    # get_wav_out_all()
    # get_wav_kde_all()
    # get_wav_lab_all_watershed_segment()
    plot_wav_watershed_segmentation_white_background()
    # plot_wav_ethogram_segment()

    # get_stp_out_all()
    # get_stp_kde_all()
    # get_stp_lab_all_watershed_segment()
    # plot_stp_watershed_segmentation_white_background()
    # plot_stp_ethogram_segment()

    # get_plot_entropies()
    # get_plot_trial_label_entropy_histograms()
    # pass
