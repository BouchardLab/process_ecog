from itertools import product
from collections import defaultdict

import h5py
import numpy
import six
import tables
from six.moves import zip, range

from fuel.datasets import Dataset
from fuel.utils import do_not_pickle_attributes


@do_not_pickle_attributes('data_sources', 'external_file_handle',
                          'source_shapes', 'fold')
class H5PYDatasetFolds(Dataset):
    """An h5py-fueled HDF5 dataset.

    This dataset class assumes a particular file layout:

    * Data sources reside in the root group, and their names define the
      source names.
    * Data sources are not explicitly split. Instead, splits are defined
      in the `split` attribute of the root group. It's expected to be a
      1D numpy array of compound ``dtype`` with seven fields, organized as
      follows:

      1. ``split`` : string identifier for the split name
      2. ``source`` : string identifier for the source name
      3. ``fold`` : int fold identified
      4. ``indices`` : h5py.Reference, reference to a dataset containing
         subset indices for this split/source/fold.
      5. ``available`` : boolean, ``False`` is this split is not available
         for this source
      6. ``comment`` : comment string

    Parameters
    ----------
    file_or_path : :class:`h5py.File` or str
        HDF5 file handle, or path to the HDF5 file.
    which_sets : iterable of str
        Which split(s) to use. If one than more split is requested,
        the provided sources will be the intersection of provided
        sources for these splits. **Note: for all splits that are
        specified as a list of indices, those indices will get sorted
        no matter what.**
    fold : int
        Which fold of data to use.
    load_in_memory : bool, optional
        Whether to load the data in main memory. Defaults to `False`.
    driver : str, optional
        Low-level driver to use. Defaults to `None`. See h5py
        documentation for a complete list of available options.
    sort_indices : bool, optional
        HDF5 doesn't support fancy indexing with an unsorted list of
        indices. In order to allow that, the dataset can sort the list
        of indices, access the data in sorted order and shuffle back
        the data in the unsorted order. Setting this flag to `True`
        (the default) will activate this behaviour. For greater
        performance, set this flag to `False`. Note that in that case,
        it is the user's responsibility to make sure that indices are
        ordered.

    Attributes
    ----------
    sources : tuple of strings
        The sources this dataset will provide when queried for data.
    provides_sources : tuple of strings
        The sources this dataset *is able to* provide for the requested
        split.
    example_iteration_scheme : :class:`.IterationScheme` or ``None``
        The iteration scheme the class uses in order to produce a stream of
        examples.
    vlen_sources : tuple of strings
        All sources provided by this dataset which have variable length.
    default_axis_labels : dict mapping string to tuple of strings
        Maps all sources provided by this dataset to their axis labels.

    """
    interface_version = '0.3'
    _ref_counts = defaultdict(int)
    _file_handles = {}

    def __init__(self, file_or_path, which_sets, fold,
                 load_in_memory=False, driver=None, sort_indices=True,
                 **kwargs):
        if isinstance(file_or_path, h5py.File):
            self.path = file_or_path.filename
            self.external_file_handle = file_or_path
        else:
            self.path = file_or_path
            self.external_file_handle = None
        which_sets_invalid_value = (
            isinstance(which_sets, six.string_types) or
            not all(isinstance(s, six.string_types) for s in which_sets))
        if which_sets_invalid_value:
            raise ValueError('`which_sets` should be an iterable of strings')
        self.which_sets = which_sets
        self.fold = fold
        self.load_in_memory = load_in_memory
        self.driver = driver
        if load_in_memory:
            self.sort_indices = False
        else:
            self.sort_indices = sort_indices

        self._parse_dataset_info()

        kwargs.setdefault('axis_labels', self.default_axis_labels)
        super(H5PYDatasetFolds, self).__init__(**kwargs)

    def _parse_dataset_info(self):
        """Parses information related to the HDF5 interface.

        In addition to verifying that the `self.which_sets` split is
        available, this method sets the following attributes:

        * `provides_sources`
        * `vlen_sources`
        * `default_axis_labels`

        """
        self._out_of_memory_open()
        handle = self._file_handle
        available_splits = self.get_all_splits(handle)
        available_folds = self.get_all_folds(handle)
        which_sets = self.which_sets
        provides_sources = None
        for split in which_sets:
            if split not in available_splits:
                raise ValueError(
                    "'{}' split is not provided by this ".format(split) +
                    "dataset. Available splits are " +
                    "{}.".format(available_splits))
            split_provides_sources = set(
                self.get_provided_sources(handle, split, self.fold))
            if provides_sources:
                provides_sources &= split_provides_sources
            else:
                provides_sources = split_provides_sources
        self.provides_sources = tuple(sorted(provides_sources))
        self.vlen_sources = self.get_vlen_sources(handle)
        self.default_axis_labels = self.get_axis_labels(handle)
        self._out_of_memory_close()

    @staticmethod
    def create_split_array(split_list):
        """Create a valid array for the `split` attribute of the root node.
        """
        split_len = max(len(split[0]) for split in split_list)
        sources = set()
        comment_len = 1
        for split in split_list:
            sources.add(split[1])
            comment_len = max([comment_len, len(split[-1])])
        sources = sorted(list(sources))
        source_len = max(len(source) for source in sources)

        # Instantiate empty split array
        split_array = numpy.empty(
            len(split_list),
            dtype=numpy.dtype([
                ('split', 'a', split_len),
                ('source', 'a', source_len),
                ('fold', numpy.int64, 1),
                ('indices', h5py.special_dtype(ref=h5py.Reference)),
                ('available', numpy.bool, 1),
                ('comment', 'a', comment_len)]))

        # Fill split array
        for i, split in enumerate(split_list):
            # Workaround for H5PY being unable to store unicode type
            split_array[i]['split'] = split[0].encode('utf8')
            split_array[i]['source'] = split[1].encode('utf8')
            split_array[i]['fold'] = split[2]
            split_array[i]['indices'] = split[3]
            split_array[i]['available'] = split[4]
            split_array[i]['comment'] = split[5].encode('utf8')

        return split_array

    @staticmethod
    def get_all_splits(h5file):
        """Returns the names of all splits of an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.

        Returns
        -------
        available_splits : tuple of str
            Names of all splits in ``h5file``.

        """
        available_splits = tuple(
            set(row['split'].decode('utf8') for row in h5file.attrs['split']))
        return available_splits

    @staticmethod
    def get_all_sources(h5file):
        """Returns the names of all sources of an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.

        Returns
        -------
        all_sources : tuple of str
            Names of all sources in ``h5file``.

        """
        all_sources = tuple(
            set(row['source'].decode('utf8') for row in h5file.attrs['split']))
        return all_sources

    @staticmethod
    def get_all_folds(h5file):
        """Returns folds HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.

        Returns
        -------
        all_folds : tuple of int 
            Folds in ``h5file``.

        """
        all_folds = tuple(
            set(row['fold'] for row in h5file.attrs['split']))
        return all_folds

    @staticmethod
    def get_provided_sources(h5file, split, fold):
        """Returns the sources provided by a specific split.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.
        split : str
            Name of the split.

        Returns
        -------
        provided_sources : tuple of str
            Names of sources provided by ``split`` in ``h5file``.

        """
        provided_sources = tuple(
            row['source'].decode('utf8') for row in h5file.attrs['split']
            if (row['split'].decode('utf8') == split)
            and (row['available'])
            and (row['fold'] == fold))
        return provided_sources

    @staticmethod
    def get_vlen_sources(h5file):
        """Returns the names of variable-length sources in an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.
        split : str
            Name of the split.

        Returns
        -------
        vlen_sources : tuple of str
            Names of all variable-length sources in ``h5file``.

        """
        vlen_sources = []
        for source_name in H5PYDatasetFolds.get_all_sources(h5file):
            source = h5file[source_name]
            if len(source.dims) > 0 and 'shapes' in source.dims[0]:
                if len(source.dims) > 1:
                    raise ValueError('Variable-length sources must have only '
                                     'one dimension.')
                vlen_sources.append(source_name)
        return vlen_sources

    @staticmethod
    def get_axis_labels(h5file):
        """Returns axis labels for all sources in an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.

        Returns
        -------
        axis_labels : dict
            Maps source names to a tuple of str representing the axis
            labels.

        """
        axis_labels = {}
        vlen_sources = H5PYDatasetFolds.get_vlen_sources(h5file)
        for source_name in H5PYDatasetFolds.get_all_sources(h5file):
            if source_name in vlen_sources:
                axis_labels[source_name] = (
                    (h5file[source_name].dims[0].label,) +
                    tuple(label.decode('utf8') for label in
                          h5file[source_name].dims[0]['shape_labels']))
            else:
                axis_labels[source_name] = tuple(
                    dim.label for dim in h5file[source_name].dims)
        return axis_labels

    @staticmethod
    def get_indices(h5file, split, fold):
        """Returns subset indices for sources of a specific split.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.
        split : str
            Name of the split.
        fold : int
            Fold of data.

        Returns
        -------
        indices : dict
            Maps source names to a list of indices. Note that only
            sources for which indices are specified appear in this dict.

        """
        indices = {}
        for row in h5file.attrs['split']:
            if (row['split'].decode('utf8') == split) and\
            (row['fold'] == fold):
                source = row['source'].decode('utf8')
                if row['indices']:
                    indices[source] = h5file[row['indices']]
        return indices

    @staticmethod
    def unsorted_fancy_index(request, indexable):
        """Safe unsorted list indexing.

        Some objects, such as h5py datasets, only support list indexing
        if the list is sorted.

        This static method adds support for unsorted list indexing by
        sorting the requested indices, accessing the corresponding
        elements and re-shuffling the result.

        Parameters
        ----------
        request : list of int
            Unsorted list of example indices.
        indexable : any fancy-indexable object
            Indexable we'd like to do unsorted fancy indexing on.

        """
        if len(request) > 1:
            indices = numpy.argsort(request)
            data = numpy.empty(shape=(len(request),) + indexable.shape[1:],
                               dtype=indexable.dtype)
            data[indices] = indexable[numpy.array(request)[indices], ...]
        else:
            data = indexable[request]
        return data

    def load(self):
        # If the dataset is unpickled, it makes no sense to have an external
        # file handle. However, since `load` is also called during the lifetime
        # of a dataset (e.g. if load_in_memory = True), we don't want to
        # accidentally overwrite the reference to a potential external file
        # handle, hence this check.
        if not hasattr(self, '_external_file_handle'):
            self.external_file_handle = None

        self._out_of_memory_open()
        handle = self._file_handle

        # Load subset slices / indices
        subsets = []
        if len(self.which_sets) > 1:
            indices = defaultdict(list)
            for split in self.which_sets:
                split_indices = self.get_indices(handle,
                                                 split,
                                                 self.fold)
                for source_name in self.sources:
                    if source_name in split_indices:
                        ind = split_indices[source_name]
                    else:
                        raise ValueError
                    indices[source_name] = sorted(
                        set(indices[source_name] + ind))
        else:
            indices = self.get_indices(handle,
                                       self.which_sets[0],
                                       self.fold)
        num_examples = None
        for source_name in self.sources:
            if source_name in indices:
                subset = indices[source_name]
            else:
                raise ValueError
            subsets.append(subset)
            subset_num_examples = len(subset)
            if num_examples is None:
                num_examples = subset_num_examples
            if num_examples != subset_num_examples:
                raise ValueError("sources have different lengths")
        self.subsets = subsets

        # Load data sources and source shapes
        if self.load_in_memory:
            data_sources = []
            source_shapes = []
            for source_name, subset in zip(self.sources, self.subsets):
                data_source = handle[source_name][list(subset)]
                data_sources.append(data_source)
                if source_name in self.vlen_sources:
                    shapes = handle[source_name].dims[0]['shapes'][subset]
                else:
                    shapes = None
                source_shapes.append(shapes)
            self.data_sources = tuple(data_sources)
            self.source_shapes = tuple(source_shapes)
        else:
            self.data_sources = None
            self.source_shapes = None

        self._out_of_memory_close()

    @property
    def num_examples(self):
        return len(self.subsets[0])

    def open(self):
        return None if self.load_in_memory else self._out_of_memory_open()

    def _out_of_memory_open(self):
        if not self._external_file_handle:
            if self.path not in self._file_handles:
                handle = h5py.File(
                    name=self.path, mode="r", driver=self.driver)
                self._file_handles[self.path] = handle
            self._ref_counts[self.path] += 1

    def close(self, state):
        if not self.load_in_memory:
            self._out_of_memory_close()

    def _out_of_memory_close(self):
        if not self._external_file_handle:
            self._ref_counts[self.path] -= 1
            if not self._ref_counts[self.path]:
                del self._ref_counts[self.path]
                self._file_handles[self.path].close()
                del self._file_handles[self.path]

    @property
    def _file_handle(self):
        if self._external_file_handle:
            return self._external_file_handle
        elif self.path in self._file_handles:
            return self._file_handles[self.path]
        else:
            raise IOError('no open handle for file {}'.format(self.path))

    def get_data(self, state=None, request=None):
        if self.load_in_memory:
            data, shapes = self._in_memory_get_data(state, request)
        else:
            data, shapes = self._out_of_memory_get_data(state, request)
        for i in range(len(data)):
            if shapes[i] is not None:
                for j in range(len(data[i])):
                    data[i][j] = data[i][j].reshape(shapes[i][j])
        return tuple(data)

    def _in_memory_get_data(self, state=None, request=None):
        if state is not None or request is None:
            raise ValueError
        data = [data_source[request] for data_source in self.data_sources]
        shapes = [shape[request] if shape is not None else None
                  for shape in self.source_shapes]
        return data, shapes

    def _out_of_memory_get_data(self, state=None, request=None):
        if not isinstance(request, (slice, list)):
            raise ValueError()
        data = []
        shapes = []
        handle = self._file_handle
        for source_name, subset in zip(self.sources, self.subsets):
            if hasattr(subset, 'step'):
                if hasattr(request, 'step'):
                    req = slice(request.start + subset.start,
                                request.stop + subset.start, request.step)
                else:
                    req = [index + subset.start for index in request]
            else:
                req = subset[request]
            if hasattr(req, 'step'):
                val = handle[source_name][req]
                if source_name in self.vlen_sources:
                    shape = handle[source_name].dims[0]['shapes'][req]
                else:
                    shape = None
            else:
                if self.sort_indices:
                    val = self.unsorted_fancy_index(req, handle[source_name])
                    if source_name in self.vlen_sources:
                        shape = self.unsorted_fancy_index(
                            req, handle[source_name].dims[0]['shapes'])
                    else:
                        shape = None
                else:
                    val = handle[source_name][req]
                    if source_name in self.vlen_sources:
                        shape = handle[source_name].dims[0]['shapes'][req]
                    else:
                        shape = None
            data.append(val)
            shapes.append(shape)
        return data, shapes
