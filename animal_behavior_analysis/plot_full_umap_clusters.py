"""Explore the clustering results, and other features."""
import sys
import config_dodo
import config
from utilities import read_pickle, write_pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

frame_rate = 100.0
latencies = {}
num_frames = {}
lat_fra_ratios = {}
for key in config_dodo.KEY_LIST:
    name = config_dodo.SUBJECT_NAME.format(*key)
    print(name)
    pickle_end = name + ".pickle"
    path_kal = config_dodo.KAL_PATH / f"kal_xy_{pickle_end}"
    kals = read_pickle(path_kal)[:, :, 1]
    # plt.plot(kals[:, 0], alpha=0.9)
    volatility = pd.DataFrame(kals).rolling(500).std().mean(axis=1) * 10
    volatility = np.gradient(savgol_filter(volatility, 501, 3)) * 500
    volatility = volatility / np.nanmax(volatility) * 100.0
    # plt.plot(volatility, alpha=0.9)
    peaks, properties = find_peaks(volatility, height=20.0, prominence=20.0)
    latencies[key] = peaks[-1] / frame_rate
    num_frames[key] = len(kals)

    # plt.plot(peaks, volatility[peaks], "x", alpha=0.9)
    # plt.vlines(
    #     x=peaks,
    #     ymin=volatility[peaks] - properties["prominences"],
    #     ymax=volatility[peaks],
    #     alpha=0.9,
    # )
    # plt.title(name)
    # sns.despine(fig=fig, trim=True)
    # plt.show()
    # plt.close(fig=fig)

sns.set_context("paper", font_scale=1.5)
plt.figure(figsize=(7, 4))
data = []
for key, value in latencies.items():
    data.append({"mouse": key[0], "day": key[1], "trial": key[2], "lat": value})
data = pd.DataFrame(data)
# sns.boxplot(x="day", y="lat", hue="trial", data=data, whis=2.0)
# sns.violinplot(x="day", y="lat", hue="trial", data=data)
# sns.violinplot(x="day", y="lat", hue="trial", data=data)
# sns.boxplot(x="day", y="lat", hue="trial", data=data, whis=2.0)
# sns.barplot(x="day", y="lat", hue="trial", data=data, capsize=0.2)
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

# for key in config_dodo.KEY_LIST:
#     name = config_dodo.SUBJECT_NAME.format(*key)
#     print(name)
#     pickle_end = name + ".pickle"
#     path_out_wav = config_dodo.OUT_PATH / f"out_wav_{pickle_end}"
#     out_wav = read_pickle(path_out_wav)
#     path_coa = config_dodo.LAB_PATH / f"lab_coa_wav_{pickle_end}"
#     (mem_coarse_wavs, lab_coarse_wavs, sco_coarse_wavs) = read_pickle(path_coa)
#     print(np.unique(lab_coarse_wavs))
#     print(mem_coarse_wavs.shape)
#     label_colors = [colors[np.argmax(scores)] for scores in mem_coarse_wavs]
#     plt.scatter(*out_wav.T, s=1, alpha=0.01, c=label_colors)
# plt.xlabel(r"UMAP$_1$")
# plt.ylabel(r"UMAP$_2$")
# plt.xlim(-15, 25)
# plt.ylim(-15, 25)
# sns.despine(fig=fig, trim=True)
# plt.savefig("full_umap_cluster.png", bbox_inches="tight", dpi=400)
# plt.close(fig=fig)


# sns.set_context("paper", font_scale=1.5)

# clu_coarse_wavs = read_pickle(config_dodo.CLU_PATH / "clu_coa_wav.pickle")
# num_labels = clu_coarse_wavs.labels_.max() + 1
# colors = cm.get_cmap("gist_rainbow", num_labels)(range(num_labels))
# fig = plt.figure(figsize=(7, 7))
# for key in config_dodo.KEY_LIST:
#     name = config_dodo.SUBJECT_NAME.format(*key)
#     print(name)
#     pickle_end = name + ".pickle"
#     path_out_wav = config_dodo.OUT_PATH / f"out_wav_{pickle_end}"
#     out_wav = read_pickle(path_out_wav)
#     path_coa = config_dodo.LAB_PATH / f"lab_coa_wav_{pickle_end}"
#     (mem_coarse_wavs, lab_coarse_wavs, sco_coarse_wavs) = read_pickle(path_coa)
#     print(np.unique(lab_coarse_wavs))
#     print(mem_coarse_wavs.shape)
#     label_colors = [colors[np.argmax(scores)] for scores in mem_coarse_wavs]
#     plt.scatter(*out_wav.T, s=1, alpha=0.01, c=label_colors)
# plt.xlabel(r"UMAP$_1$")
# plt.ylabel(r"UMAP$_2$")
# plt.xlim(-15, 25)
# plt.ylim(-15, 25)
# sns.despine(fig=fig, trim=True)
# plt.savefig("full_umap_cluster.png", bbox_inches="tight", dpi=400)
# plt.close(fig=fig)


# sys.exit()

# MY_VIDEOS = [(297, 1, 3), (297, 5, 5), (298, 1, 1), (298, 5, 4)]


# # labels = np.random.random(size=(10, 100))
# # num_labels = 10
# # labels = np.random.choice(num_labels, size=(100))


# path_coa = config_dodo.LAB_PATH / f"lab_coa_wav_{pickle_end}"
# (mem_coarse_wavs, lab_coarse_wavs, sco_coarse_wavs) = read_pickle(path_coa)
# # path_fin = config_dodo.LAB_PATH / f"lab_fin_wav_{pickle_end}"
# # (mem_fine_wavs, lab_fine_wavs, sco_fine_wavs) = read_pickle(path_fin)

# labels = [np.argmax(scores) for scores in mem_coarse_wavs]
# # labels = lab_coarse_wavs
# clu_coarse_wavs = read_pickle(config_dodo.CLU_PATH / "clu_coa_wav.pickle")
# num_labels = clu_coarse_wavs.labels_.max() + 1

# # labels = [np.argmax(scores) for scores in mem_fine_wavs]
# # # labels = lab_fine_wavs
# # clu_fine_wavs = read_pickle(config_dodo.CLU_PATH / "clu_fin_wav.pickle")
# # num_labels = clu_fine_wavs.labels_.max() + 1

# events = [
#     [i for i, l in enumerate(labels) if l == label]
#     for label in range(num_labels)
#     # [i for i, l in enumerate(labels) if l == label]
#     # for label in range(-1, num_labels)
# ]
# colors = cm.get_cmap("gist_rainbow", num_labels)(range(num_labels))
# # colors = np.concatenate(([[0.5, 0.5, 0.5, 1.0]], colors))
# fig = plt.figure(figsize=(7, 7))
# plt.eventplot(events, colors=colors)  # , lineoffsets=0.0)
# sns.despine(fig=fig, trim=True)
# plt.show()

# # x = np.linspace(-2, 2, 200)

# # duration = 2

# # fig, ax = plt.subplots()


# # def make_frame(t):
# #     ax.clear()
# #     ax.plot(x, np.sinc(x ** 2) + np.sin(x + 2 * np.pi / duration * t), lw=3)
# #     ax.set_ylim(-1.5, 2.5)
# #     return mplfig_to_npimage(fig)


# # animation = VideoClip(make_frame, duration=duration)
# # animation.write_gif(config_dodo.ABS_PATH / "matplotlib.gif", fps=20)
