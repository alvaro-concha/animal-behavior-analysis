"""Embed new data in a previously trained UMAP."""
import config_dodo
from utilities import read_pickle, write_pickle
import config
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import umap


dep_pickle_paths = {
    "ang": [],
    "wav": [],
    "stat": config_dodo.ANG_PATH / "stat_global.pickle",
    "pca": config_dodo.WAV_PATH / "pca_fit_wav.pickle",
    "emb_ang": config_dodo.EMB_PATH / "emb_ang_sample.pickle",
    "emb_wav": config_dodo.EMB_PATH / "emb_wav_sample.pickle",
}
target_pickle_paths = {
    "out_ang": [],
    "out_wav": [],
}
for key in config_dodo.KEY_LIST:
    name = config_dodo.SUBJECT_NAME.format(*key)
    pickle_end = name + ".pickle"
    dep_pickle_paths["ang"].append(config_dodo.ANG_PATH / f"ang_{pickle_end}")
    dep_pickle_paths["wav"].append(config_dodo.WAV_PATH / f"wav_{pickle_end}")
    target_pickle_paths["out_ang"].append(
        config_dodo.OUT_PATH / f"out_ang_{pickle_end}"
    )
    target_pickle_paths["out_wav"].append(
        config_dodo.OUT_PATH / f"out_wav_{pickle_end}"
    )


# emb_angs = read_pickle(dep_pickle_paths["emb_ang"])
# embedding = emb_angs.embedding_

# emb_wavs = read_pickle(dep_pickle_paths["emb_wav"])
# embedding = emb_wavs.embedding_

# plt.scatter(*embedding.T)
# plt.show()

# import datashader as ds, pandas as pd, colorcet
# cvs = ds.Canvas(plot_width=850, plot_height=500)
# agg = cvs.points(df, 'longitude', 'latitude')
# img = ds.tf.shade(agg, cmap=colorcet.fire, how='log')

# import holoviews as hv
# import holoviews.operation.datashader as hd

# hd.shade.cmap = ["lightblue", "darkblue"]
# hv.extension("bokeh", "matplotlib")

# import pandas as pd
# import numpy as np
# import datashader as ds
# import datashader.transfer_functions as tf
# from collections import OrderedDict as odict

# num = 100000
# np.random.seed(1)

# dists = {
#     cat: pd.DataFrame(
#         odict(
#             [
#                 ("x", np.random.normal(x, s, num)),
#                 ("y", np.random.normal(y, s, num)),
#                 ("val", val),
#                 ("cat", cat),
#             ]
#         )
#     )
#     for x, y, s, val, cat in [
#         (2, 2, 0.03, 10, "d1"),
#         (2, -2, 0.10, 20, "d2"),
#         (-2, -2, 0.50, 30, "d3"),
#         (-2, 2, 1.00, 40, "d4"),
#         (0, 0, 3.00, 50, "d5"),
#     ]
# }

# df = pd.concat(dists, ignore_index=True)
# df["cat"] = df["cat"].astype("category")

# points = hv.Points(df.sample(10000))

# hv.output(backend="bokeh")
# hd.datashade(points)

import holoviews as hv
import numpy as np
from holoviews.operation.datashader import datashade, dynspread

hv.extension("bokeh")
# hv.extension("plotly")

#### Disable dynamic updating of plot
# datashade.dynamic = False

np.random.seed(1)

positions = np.random.multivariate_normal((0, 0), [[0.1, 0.1], [0.1, 1.0]], (1000,))
points = hv.Points(positions, label="Points")

# points = hv.Points(embedding, label="Points")

# plot = datashade(points, dynamic=False) + dynspread(datashade(points))
# plot = datashade(points, dynamic=False) + dynspread(datashade(points))
plot = datashade(points)

# renderer = hv.renderer("matplotlib").instance(fig="svg")
# renderer = hv.renderer("plotly").instance(fig="html")
renderer = hv.renderer("bokeh").instance(fig="html")
renderer.save(plot, "testing_bokeh")
