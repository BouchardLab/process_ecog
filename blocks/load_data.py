import numpy as np

from fuel.datasets import IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from pylearn2.datasets import ecog

def pylearn2_ecog(audio, batch_size, kwargs):
    splits = ['train', 'valid', 'test']
    datasets = []
    if audio:
        raise NotImplementedError
    else:
        data_file = '${PYLEARN2_DATA_PATH}/ecog/EC2_CV_85_nobaseline_aug.h5'
    for sp in splits:
        cur_dataset = ecog.ECoG(data_file, sp, **kwargs)
        ds_x = np.squeeze(cur_dataset.get_topological_view())
        ds_y = cur_dataset.y[:,np.newaxis,:]
        d = {'x': ds_x,
             'y': ds_y}
        datasets.append(IndexableDataset(d))

    streams = [DataStream(ds,
        iteration_scheme=SequentialScheme(ds.num_examples, batch_size)) for ds in datasets]
    return zip(splits, streams)
