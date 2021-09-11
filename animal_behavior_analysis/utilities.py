"""Common utility functions.

Write and read pickles.
"""
import pickle

################################## FUNCTIONS ###################################


def write_pickle(obj, path):
    """
    Writes an object as a pickle file.

    Parameters
    ----------
    obj : pickable object
        Object to be pickled
    path : pathlib.Path
        Path to pickle file to be saved
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file, protocol=-1, fix_imports=False)


def read_pickle(path):
    """
    Loads an object from a pickle file.

    Parameters
    ----------
    path : pathlib.Path
        Path to pickle file to be opened

    Returns
    -------
    obj : pickable object
        Unpickled object
    """
    with path.open("rb") as file:
        return pickle.load(file)
