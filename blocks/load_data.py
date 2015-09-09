import numpy as np

from fuel.datasets import IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from pylearn2.datasets import ecog

def pylearn2_ecog(ds_params, static_params):
    splits = ['train', 'valid', 'test']
    datasets = []
    audio = static_params['audio']
    batch_size = static_params['monitor_batch_size']
    if audio:
        raise NotImplementedError
    else:
        data_file = '${PYLEARN2_DATA_PATH}/ecog/EC2_CV_85_nobaseline_aug.h5'
    for sp in splits:
        cur_dataset = ecog.ECoG(data_file, sp, **ds_params)
        ds_x = np.squeeze(cur_dataset.get_topological_view())
        ds_y = cur_dataset.y.argmax(axis=1).astype(np.int32)
        d = {'x': ds_x,
             'y': ds_y}
        datasets.append(IndexableDataset(d))

    streams = [DataStream(ds,
        iteration_scheme=SequentialScheme(ds.num_examples, batch_size)) for ds in datasets]
    return zip(splits, streams)
