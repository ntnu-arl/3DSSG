import torch.utils.data as data
from torch_geometric.data import HeteroData
from pathlib import Path
import os
import torch
import gzip
import pickle
import numpy as np
import torch
import copy
from typing import Tuple, List
import itertools
import logging
import open3d as o3d

from reasoning_ros_interface.methods.ssg_3d.ssg import define
from reasoning_ros_interface.methods.ssg_3d.ssg.utils import util_data
from reasoning_ros_interface.methods.ssg_3d.codeLib import transformation
from reasoning_ros_interface.methods.ssg_3d.codeLib.utils.util import read_txt_to_list

logger_py = logging.getLogger(__name__)



class OakDataset(data.Dataset):
    def __init__(self, config, mode, **args):
        super().__init__()
        assert mode in ['train', 'validation', 'test']
        self._device = config.DEVICE
        path = config.data['path']
        self.config = config
        self.cfg = self.config
        self.mconfig = config.data
        self.path = Path(config.data.path)
        self.multi_rel_outputs = multi_rel_outputs = config.model.multi_rel
        self.use_rgb = config.model.use_rgb
        self.use_normal = config.model.use_normal
        
        self.scans = os.listdir(path)
        
        self.size = len(self.scans)
        
        pth_classes = os.path.join('/media/albert/ExtremeAlbert/scene_graphs_datasets/3RScan/3DSSG_subset', 'classes.txt')
        pth_relationships = os.path.join('/media/albert/ExtremeAlbert/scene_graphs_datasets/3RScan/3DSSG_subset', 'relationships.txt')
        names_classes = read_txt_to_list(pth_classes)
        names_relationships = read_txt_to_list(pth_relationships)
        if not multi_rel_outputs:
            if define.NAME_NONE not in names_relationships:
                names_relationships.append(define.NAME_NONE)
        elif define.NAME_NONE in names_relationships:
            names_relationships.remove(define.NAME_NONE)       
        self.relationNames = sorted(names_relationships)
        self.classNames = sorted(names_classes)
        
    def load_map(self, path: Path, normals_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                                torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        with gzip.open(path, "rb") as f:
            results = pickle.load(f)['objects']
        pcl = []
        mask = []
        bbox = []
        rgb = []
        instances = []
        descriptors = []
        normals = []
        
        all_normals = np.load(normals_path)
        
        current = 0
        for i, result in enumerate(results):
            new_obj = copy.deepcopy(result)
            if self.mconfig.node_feature_dim == -1:
                pcl.append(torch.from_numpy(new_obj['pcd_np']))
                rgb.append(torch.from_numpy(new_obj['pcd_color_np']))
                normals = torch.from_numpy(all_normals)
            else:
                choice = np.random.choice(len(new_obj['pcd_np']), self.mconfig.node_feature_dim, 
                                          replace=len(new_obj['pcd_np']) < self.mconfig.node_feature_dim)
                pcl.append(torch.from_numpy(new_obj['pcd_np'][choice]))
                rgb.append(torch.from_numpy(new_obj['pcd_color_np'][choice]))
                normals.append(torch.from_numpy(all_normals[choice + current]))
                current += len(new_obj['pcd_np'])
                
            mask.append(i * torch.ones((new_obj['pcd_np'].shape[0], 1)))
            bbox.append(torch.from_numpy(new_obj['bbox_np']))
            instances.append(new_obj['class_name'])
            descriptors.append(util_data.gen_descriptor_pts(pcl[i]))
            pcl[i] = self.norm_tensor(pcl[i])
            
            del new_obj['pcd_np']
            del new_obj['pcd_color_np']
            del new_obj['bbox_np']
        
        # Step 1: Create a dictionary mapping each unique string to a unique integer
        class_to_idx = {name: idx for idx, name in enumerate(sorted(set(instances)))}

        # Step 2: Map the list of strings to their corresponding integers
        class_ids = [class_to_idx[name] for name in instances]

        # Step 3: Convert the list of integers to a PyTorch tensor of type long
        tensor_of_classes = torch.tensor(class_ids, dtype=torch.long)
            
        mask = torch.cat(mask, dim=0).long()
        bbox = torch.stack(bbox, dim=0).float()
        descriptors = torch.stack(descriptors, dim=0).float()
        pcl = torch.stack(pcl, dim=0).float()
        rgb = torch.stack(rgb, dim=0).float()
        normals = torch.stack(normals, dim=0).float()
        #torch.Size([13, 9, 256])
        return pcl, mask, rgb, normals, bbox, descriptors, tensor_of_classes, instances
        
    def __getitem__(self, index):
        
        output = HeteroData()
        output['scan_id'] = self.scans[index]
        
        points, mask, rgb, normals, bbox, descriptors, classes, instances = self.load_map(self.path / output['scan_id'] / 'exps/r_mapping_stride1/pcd_r_mapping_stride1.pkl.gz',
                                                                                          self.path / output['scan_id'] / 'normals.npy')
        num_instances = len(instances)
        output['node'].x = torch.zeros([num_instances, 1])
        output['node'].y = torch.zeros([num_instances, 1]) # GT labels, in out case we don't have them
        output['node'].oid = torch.arange(num_instances).long()
        
        if self.mconfig.load_points:
            output['node'].pts = points
            if self.use_rgb:
                output['node'].pts = torch.cat([output['node'].pts, rgb], dim=-1)
            if self.use_normal:
                output['node'].pts = torch.cat([output['node'].pts, normals], dim=-1)
            if 'edge_desc' not in self.mconfig or self.mconfig['edge_desc'] == 'pts':
                output['node'].desp = descriptors
        output['node'].pts = output['node'].pts.permute(0, 2, 1)
        output['node', 'to', 'node'].edge_index = torch.tensor(list(itertools.permutations(output['node'].oid.tolist(), 2))).t().contiguous().long()
        output['node', 'to', 'node'].y = torch.zeros(output['node', 'to', 'node'].edge_index.shape[1]).long()
        
        return output
    
    def __len__(self):
        return self.size
    
    def norm_tensor(self, points):
        assert points.ndim == 2
        assert points.shape[1] == 3
        centroid = torch.mean(points, dim=0)  # N, 3
        points -= centroid  # n, 3, npts
        # find maximum distance for each n -> [n]
        furthest_distance = points.pow(2).sum(1).sqrt().max()
        points /= furthest_distance
        return points
    
    def data_augmentation(self, points):
        # random rotate
        matrix = np.eye(3)
        matrix[0:3, 0:3] = transformation.rotation_matrix(
            [0, 0, 1], np.random.uniform(0, 2*np.pi, 1))
        centroid = points[:, :3].mean(0)
        points[:, :3] -= centroid
        points[:, :3] = np.dot(points[:, :3], matrix.T)
        if self.use_normal:
            ofset = 3
            if self.use_rgb:
                ofset += 3
            points[:, ofset:3 +
                   ofset] = np.dot(points[:, ofset:3+ofset], matrix.T)
            
        return points
        