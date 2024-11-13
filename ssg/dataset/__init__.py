from .dataloader_SGFN import SGFNDataset
from . import dataloader_SGFN_seq
from .dataloader_oak import OakDataset
from .dataloader_pcd import PCDDataset
dataset_dict = {
  'sgfn': SGFNDataset,
   'sgfn_seq': dataloader_SGFN_seq.Dataset,
   'oak': OakDataset,
   'pcd': PCDDataset
}

# __all__ = ['dataset_dict']

