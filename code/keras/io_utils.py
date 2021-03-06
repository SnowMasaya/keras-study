"""Utilities related to disk I/O."""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import sys
from collections import defaultdict

try:
    import h5py
except ImportError:
    h5py = None

try:
    import tables
except ImportError:
    tables = None


class HDF5Matrix(object):
    """Representation of HDF5 dataset to be used instead of a Numpy array.

    # Example

    ```python
        x_data = HDF5Matrix('input/file.hdf5', 'data')
        model.predict(x_data)
    ```

    Providing `start` and `end` allows use of a slice of the dataset.

    Optionally, a normalizer function (or lambda) can be given. This will
    be called on every slice of data retrieved.

    # Arguments
        datapath: string, path to a HDF5 file
        dataset: string, name of the HDF5 dataset in the file specified
            in datapath
        start: int, start of desired slice of the specified dataset
        end: int, end of desired slice of the specified dataset
        normalizer: function to be called on data when retrieved

    # Returns
        An array-like HDF5 dataset.
    """
    refs = defaultdict(int)

    def __init__(self, datapath, dataset, start=0, end=None, normalizer=None):
        if h5py is None:
            raise ImportError('The use of HDF5Matrix requires '
                              'HDF5 and h5py installed.')

        if datapath not in list(self.refs.keys()):
            f = h5py.File(datapath)
            self.refs[datapath] = f
        else:
            f = self.refs[datapath]
        self.data = f[dataset]
        self.start = start
        if end is None:
            self.end = self.data.shape[0]
        else:
            self.end = end
        self.normalizer = normalizer

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.stop + self.start <= self.end:
                idx = slice(key.start + self.start, key.stop + self.start)
            else:
                raise IndexError
        elif isinstance(key, int):
            if key + self.start < self.end:
                idx = key + self.start
            else:
                raise IndexError
        elif isinstance(key, np.ndarray):
            if np.max(key) + self.start < self.end:
                idx = (self.start + key).tolist()
            else:
                raise IndexError
        elif isinstance(key, list):
            if max(key) + self.start < self.end:
                idx = [x + self.start for x in key]
            else:
                raise IndexError
        if self.normalizer is not None:
            return self.normalizer(self.data[idx])
        else:
            return self.data[idx]

    @property
    def shape(self):
        return (self.end - self.start,) + self.data.shape[1:]


class OFFMatrix(object):
    """Representation of OFF dataset to be used instead of a Numpy array.

    # Example

    ```python
        x_data = OFFMatrix('input/file.off', 'data')
        model.predict(x_data)
    ```

    Providing `start` and `end` allows use of a slice of the dataset.

    Optionally, a normalizer function (or lambda) can be given. This will
                                                be called on every slice of data retrieved.

    # Arguments
        datapath: string, path to a OFF file
        dataset: string, name of the OFF dataset in the file specified
            in datapath
        start: int, start of desired slice of the specified dataset
        end: int, end of desired slice of the specified dataset
        normalizer: function to be called on data when retrieved

    # Returns
        An array-like OFF dataset.
    """

    def __init__(self, datapath, dataset, start=0, end=None, normalizer=None):
        try:
            with open(datapath + dataset, "r") as f:
                self.read_data = f.read().strip().split()
        except IOError as io:
            print(str(io))
        self.data = {}
        self.data["file_format"] = self.read_data.pop(0)
        self.data["num_vertices"] = int(self.read_data.pop(0))
        self.data["num_faces"] = int(self.read_data.pop(0))
        self.data["num_edges"] = int(self.read_data.pop(0))
        VERTICES_DIM = 3
        FACES_DIM = 4
        if self.data["num_vertices"] != 0:
            self.data["vertices"] = self.__get_data_process(self.data["num_vertices"], VERTICES_DIM)
            print(self.data["vertices"])
        if self.data["num_faces"] != 0:
            self.data["faces"] = self.__get_data_process(self.data["num_faces"], FACES_DIM)
            print(self.data["faces"])

    def __get_data_process(self, data_number, points):
        tmp_list = []
        data_array = []
        for i in range(data_number):
            [tmp_list.append(self.read_data.pop(0)) for j in range(points)]
            data_array.append(tmp_list)
            tmp_list = []
        return np.array(data_array)

    def __len__(self):
        pass

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        else:
            raise IndexError

    @property
    def shape(self):
        pass


def save_array(array, name):
    if tables is None:
        raise ImportError('The use of `save_array` requires '
                          'the tables module.')
    f = tables.open_file(name, 'w')
    atom = tables.Atom.from_dtype(array.dtype)
    ds = f.create_carray(f.root, 'data', atom, array.shape)
    ds[:] = array
    f.close()


def load_array(name):
    if tables is None:
        raise ImportError('The use of `load_array` requires '
                          'the tables module.')
    f = tables.open_file(name)
    array = f.root.data
    a = np.empty(shape=array.shape, dtype=array.dtype)
    a[:] = array[:]
    f.close()
    return a


def ask_to_proceed_with_overwrite(filepath):
    """Produces a prompt asking about overwriting a file.

    # Arguments
        filepath: the path to the file to be overwritten.

    # Returns
        True if we can proceed with overwrite, False otherwise.
    """
    get_input = input
    if sys.version_info[:2] <= (2, 7):
        get_input = raw_input
    overwrite = get_input('[WARNING] %s already exists - overwrite? '
                          '[y/n]' % (filepath))
    while overwrite not in ['y', 'n']:
        overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
    if overwrite == 'n':
        return False
    print('[TIP] Next time specify overwrite=True!')
    return True
