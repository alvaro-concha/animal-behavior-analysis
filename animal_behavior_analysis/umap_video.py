"""UMAP Videos"""
import sys
import numpy as np
import config_dodo
from utilities import read_pickle
from cv2 import perspectiveTransform
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
import seaborn as sns

m, d, t, fl, fr, fb, ft, bl, br, bb, bt = sys.argv[1:]
fl, fr, fb, ft, bl, br, bb, bt = (
    float(fl),
    float(fr),
    float(fb),
    float(ft),
    float(bl),
    float(br),
    float(bb),
    float(bt),
)
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

out_wav = read_pickle(config_dodo.OUT_PATH / "out_wav_sample.pickle")
out_wav_all = read_pickle(config_dodo.OUT_PATH / "out_wav_all.pickle")
label_wav_all = read_pickle(config_dodo.LAB_PATH / "lab_wav_all.pickle")
out_stp = read_pickle(config_dodo.OUT_PATH / "out_stp_sample.pickle")
out_stp_all = read_pickle(config_dodo.OUT_PATH / "out_stp_all.pickle")
label_stp_all = read_pickle(config_dodo.LAB_PATH / "lab_stp_all.pickle")
long_data = read_pickle(config_dodo.LAB_PATH / "label_long_data.pickle")
name = config_dodo.SUBJECT_NAME.format(m, d, t)
pickle_end = name + ".pickle"
latency = read_pickle(config_dodo.MTR_PATH / f"Latency/latency_{pickle_end}")
xys = read_pickle(config_dodo.QNT_PATH / f"qnt_xy_{pickle_end}")[:latency]
front_matrix = read_pickle(config_dodo.MED_PATH / f"perspective_{name}C1.pickle")
back_matrix = read_pickle(config_dodo.MED_PATH / f"perspective_{name}C2.pickle")
inv_front_matrix = np.linalg.inv(front_matrix)
inv_back_matrix = np.linalg.inv(back_matrix)
xys_front = perspectiveTransform(xys, inv_front_matrix)
xys_back = perspectiveTransform(xys, inv_back_matrix)

fig, axs = plt.subplot_mosaic(
    [["front", "back", "wav"], ["front", "back", "stp"]], figsize=(13.65, 7)
)
for ax in axs.values():
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
axs["wav"].scatter(*out_wav[::20].T, c="0.85", s=5, alpha=0.125)
axs["wav"].axis("equal")
axs["stp"].scatter(*out_stp[::40].T, c="0.85", s=5, alpha=0.25)
axs["stp"].axis("equal")
color_markers = plt.cm.Pastel1(np.linspace(0, 1, 9))[:7][::-1]
see = long_data.query("mouse == @m and day == @d and trial == @t")
trail = 10000
alphas = np.linspace(0, 1, trail + 1) ** 2
sizes = np.linspace(0, 1, trail + 1) ** 2 * 40
label_wav = label_wav_all[see.index]
color_wav_dict = dict(zip(range(1, 11), plt.cm.tab10.colors))
color_wav = np.array(np.vectorize(color_wav_dict.get)(label_wav)).T
(circle_wav,) = axs["wav"].plot(
    [],
    [],
    "ko",
    markeredgewidth=2.25,
    markersize=10,
    markerfacecolor="none",
    markeredgecolor="k",
    zorder=30,
)
label_stp = label_stp_all[see.index]
color_stp_dict = dict(zip(range(1, 11), plt.cm.tab10.colors))
color_stp = np.array(np.vectorize(color_stp_dict.get)(label_stp)).T
(circle_stp,) = axs["stp"].plot(
    [],
    [],
    "ko",
    markeredgewidth=2.25,
    markersize=10,
    markerfacecolor="none",
    markeredgecolor="k",
    zorder=30,
)
norm = Normalize(vmin=0.5, vmax=10.5)
mappable = ScalarMappable(norm=norm, cmap="tab10")
mappable.set_array(range(1, 11))
cbar_label_wav = r"$Label$ $wavelet$"
cbar_wav = plt.colorbar(mappable, ax=axs["wav"], pad=0.0)
cbar_wav.set_label(label=cbar_label_wav, labelpad=10)
cbar_wav.set_ticks(range(1, 11))
cbar_wav.set_ticklabels(range(1, 11))
cbar_wav.ax.set_frame_on(False)
cbar_wav.ax.tick_params(axis="both", which="both", length=0)
cbar_label_stp = r"$Label$ pasos y poses"
cbar_stp = plt.colorbar(mappable, ax=axs["stp"], pad=0.0)
cbar_stp.set_label(label=cbar_label_stp, labelpad=10)
cbar_stp.set_ticks(range(1, 11))
cbar_stp.set_ticklabels(range(1, 11))
cbar_stp.ax.set_frame_on(False)
cbar_stp.ax.tick_params(axis="both", which="both", length=0)
rect_wav = Rectangle((1, 0.55), 8.5, 1, fc="none", ec="k", lw=2, zorder=20)
cbar_wav.ax.add_patch(rect_wav)
rect_stp = Rectangle((1, 0.55), 8.5, 1, fc="none", ec="k", lw=2, zorder=20)
cbar_stp.ax.add_patch(rect_stp)
plt.subplots_adjust(left=-0.01, bottom=0, right=0.98, top=1, wspace=-0.025, hspace=0.1)


def update(frame):
    path_front = (
        config_dodo.VID_PATH / f"M{m}D{d}T{t}C1" / f"video-frame{frame + 1:05d}.png"
    )
    img_front = Image.open(path_front)
    width, height = img_front.size
    left = int(fl * width)
    right = int(fr * width)
    bottom = int(fb * height)
    top = int(ft * height)
    img_front = img_front.crop((left, bottom, right, top))
    im_front = axs["front"].imshow(img_front)
    scatter_front = axs["front"].scatter(
        *(xys_front[frame, [16]] - (left, bottom)).T,
        color=color_markers[0],
        s=100,
        linewidths=2,
        marker="s",
        edgecolors="k",
        alpha=0.5,
        zorder=15,
    )
    axs["front"].set_xlim(-0.5, right - left - 0.5)
    axs["front"].set_ylim(top - bottom - 0.5, -0.5)
    path_back = (
        config_dodo.VID_PATH / f"M{m}D{d}T{t}C2" / f"video-frame{frame + 1:05d}.png"
    )
    img_back = Image.open(path_back)
    width, height = img_back.size
    left = int(bl * width)
    right = int(br * width)
    bottom = int(bb * height)
    top = int(bt * height)
    img_back = img_back.crop((left, bottom, right, top))
    im_back = axs["back"].imshow(img_back)
    scatter_back = axs["back"].scatter(
        *(xys_back[frame, [4, 17, 2, 3, 0, 1]] - (left, bottom)).T,
        color=color_markers[1:],
        s=100,
        linewidths=2,
        marker="s",
        edgecolors="k",
        alpha=0.5,
        zorder=15,
    )
    axs["back"].set_xlim(-0.5, right - left - 0.5)
    axs["back"].set_ylim(top - bottom - 0.5, -0.5)
    scatter_wav = axs["wav"].scatter(
        *out_wav_all[see.index][np.maximum(0, frame - trail) : frame + 1].T,
        c=np.column_stack(
            (
                color_wav[np.maximum(0, frame - trail) : frame + 1],
                alphas[-np.minimum(frame, trail) - 1 :],
            )
        ),
        s=sizes[-np.minimum(frame, trail) - 1 :],
        zorder=15,
    )
    circle_wav.set_data(*out_wav_all[see.index][frame])
    scatter_stp = axs["stp"].scatter(
        *out_stp_all[see.index][np.maximum(0, frame - trail) : frame + 1].T,
        c=np.column_stack(
            (
                color_stp[np.maximum(0, frame - trail) : frame + 1],
                alphas[-np.minimum(frame, trail) - 1 :],
            )
        ),
        s=sizes[-np.minimum(frame, trail) - 1 :],
        zorder=15,
    )
    circle_stp.set_data(*out_stp_all[see.index][frame])
    rect_wav.set_xy((1, int(label_wav[frame]) - 1 + 0.55))
    rect_stp.set_xy((1, int(label_stp[frame]) - 1 + 0.55))
    return [scatter_wav, scatter_stp, scatter_front, scatter_back, im_front, im_back]


(config_dodo.VID_PATH / f"M{m}D{d}T{t}").mkdir(parents=True, exist_ok=True)
for frame in range(0, len(see), 4):
    scatters = update(frame)
    path_save = (
        config_dodo.VID_PATH / f"M{m}D{d}T{t}" / f"video-frame{frame + 1:05d}.png"
    )
    plt.savefig(path_save, dpi=100)
    for scatter in scatters:
        scatter.remove()
