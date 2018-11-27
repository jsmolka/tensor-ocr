from os.path import dirname, join, realpath


def data_path(fname):
    """
    Gets a path for a given filename. This ensures that relative filenames to
    data files can be used from all modules.

    model.json -> .../src/data/model.json
    """
    return join(dirname(realpath(__file__)), fname)
