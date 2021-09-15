"""Explore the clustering results, and other features."""
import config_dodo
import config
from utilities import read_pickle, write_pickle

TO_PLOT = """
Eventplots para cada trial. Usando cada clusterer (Fine y coarse) y
usando labels (con noise) y membership vectors (sin noise).
UMAP sample plot. Usando cada clusterer y cada labeler.
Histogramas de labels para cada trial (usando cada clusterer y cada labeler).
PDF UMAP para cada trial.
Graficar entropía de las distribuciones (Histogramas de labels y PDF UMAP)
para cada trial -> Agregar todos los ratones y hacer plot Entropía vs Day-Trial.
Ver cómo combinar el Fine con el Coarse Clusterer, para romper los clusters grandes
(como el label 0 del coarse) en pedacitos más pequeños.
Para ello sería util ver los histogramas con las distribuciones marginales de labels.
"""

LEO_SAYS = """
Hola, estuve viendo y por ejemplo a simple vista se ve un cambio de comportamiento
en la variacion de movimientos tanto de las patitas como del cuerpo en general
(tanto en camara frontal como trasera) en el raton Dia1_ID298_trial1_170521;
comparado con el trial Dia5_ID298_trial4_210521, en donde el raton es mucho mas
estable en el movimiento del cuerpo y la alternancia de patitas es mas rapida y constante
(anque hay tramos en velocidades altas que puede perder el equilibrio por momentos).
En principio a nivel de angulos no deberia de haber tanta variacion en el
Trial 4 del dia 5, comparado con el dia 1 Trial 1.
Y por ahi otro que se podria analizar, es este raton Dia1_ID297_trial3_170521, que ya
desde el primer dia tuvo una performance muy alta y  compararlo por ejemplo con el trial
Dia5_ID297_trial5_210521, que para mi la performance a nivel de comportamiento se ve muy
parecida, asi que no deberia haber mucha variabilidad porque ya de entrada supo utlizar
una estrategia y la mantuvo.
"""

MY_VIDEOS = [(297, 1, 3), (297, 5, 5), (298, 1, 1), (298, 5, 4)]

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

sns.set_context("paper", font_scale=1.5)

# labels = np.random.random(size=(10, 100))
# num_labels = 10
# labels = np.random.choice(num_labels, size=(100))

name = config_dodo.SUBJECT_NAME.format(*MY_VIDEOS[0])
pickle_end = name + ".pickle"
path_out_wav = config_dodo.OUT_PATH / f"out_wav_{pickle_end}"
out_wav = read_pickle(path_out_wav)
fig = plt.figure(figsize=(7, 7))
plt.scatter(*out_wav.T, s=1, alpha=0.1)
plt.xlabel(r"UMAP$_1$")
plt.ylabel(r"UMAP$_2$")
plt.xlim(-10, 25)
plt.ylim(-15, 25)
sns.despine(fig=fig, trim=True)
plt.show()


path_coa = config_dodo.LAB_PATH / f"lab_coa_wav_{pickle_end}"
(mem_coarse_wavs, lab_coarse_wavs, sco_coarse_wavs) = read_pickle(path_coa)
# path_fin = config_dodo.LAB_PATH / f"lab_fin_wav_{pickle_end}"
# (mem_fine_wavs, lab_fine_wavs, sco_fine_wavs) = read_pickle(path_fin)

labels = [np.argmax(scores) for scores in mem_coarse_wavs]
# labels = lab_coarse_wavs
clu_coarse_wavs = read_pickle(config_dodo.CLU_PATH / "clu_coa_wav.pickle")
num_labels = clu_coarse_wavs.labels_.max() + 1

# labels = [np.argmax(scores) for scores in mem_fine_wavs]
# # labels = lab_fine_wavs
# clu_fine_wavs = read_pickle(config_dodo.CLU_PATH / "clu_fin_wav.pickle")
# num_labels = clu_fine_wavs.labels_.max() + 1

events = [
    [i for i, l in enumerate(labels) if l == label]
    for label in range(num_labels)
    # [i for i, l in enumerate(labels) if l == label]
    # for label in range(-1, num_labels)
]
colors = cm.get_cmap("gist_rainbow", num_labels)(range(num_labels))
# colors = np.concatenate(([[0.5, 0.5, 0.5, 1.0]], colors))
fig = plt.figure(figsize=(7, 7))
plt.eventplot(events, colors=colors)  # , lineoffsets=0.0)
sns.despine(fig=fig, trim=True)
plt.show()

# x = np.linspace(-2, 2, 200)

# duration = 2

# fig, ax = plt.subplots()


# def make_frame(t):
#     ax.clear()
#     ax.plot(x, np.sinc(x ** 2) + np.sin(x + 2 * np.pi / duration * t), lw=3)
#     ax.set_ylim(-1.5, 2.5)
#     return mplfig_to_npimage(fig)


# animation = VideoClip(make_frame, duration=duration)
# animation.write_gif(config_dodo.ABS_PATH / "matplotlib.gif", fps=20)
