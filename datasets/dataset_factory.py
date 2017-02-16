from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import ade20k

datasets_map = {
    'ade20k': ade20k,
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
  if name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % name)

  return datasets_map[name].get_split(
      split_name,
      dataset_dir,
      file_pattern,
      reader)
