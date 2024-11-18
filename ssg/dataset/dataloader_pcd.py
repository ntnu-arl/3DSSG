import torch.utils.data as data
from torch_geometric.data import HeteroData
from pathlib import Path
import os
import torch
import numpy as np
import torch
from typing import Dict, Tuple
import itertools
import logging
import open3d as o3d

from reasoning_ros_interface.methods.ssg_3d.ssg import define
from reasoning_ros_interface.methods.ssg_3d.ssg.utils import util_data
from reasoning_ros_interface.methods.ssg_3d.codeLib import transformation
from reasoning_ros_interface.methods.ssg_3d.codeLib.utils.util import read_txt_to_list

logger_py = logging.getLogger(__name__)



class PCDDataset(data.Dataset):
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
        self.for_eval = mode != 'train'
        
        self.scans = os.listdir(self.path)
        
        self.size = len(self.scans)
        
        classes_relations_dir = Path(__file__).resolve().parent.parent.parent / config.data.classes_relations_dir
        pth_classes = os.path.join(classes_relations_dir, 'classes.txt')
        pth_relationships = os.path.join(classes_relations_dir, 'relationships.txt')
        names_classes = read_txt_to_list(pth_classes)
        names_relationships = read_txt_to_list(pth_relationships)
        if not multi_rel_outputs:
            if define.NAME_NONE not in names_relationships:
                names_relationships.append(define.NAME_NONE)
        elif define.NAME_NONE in names_relationships:
            names_relationships.remove(define.NAME_NONE)       
        self.relationNames = sorted(names_relationships)
        self.classNames = sorted(names_classes)
        
    @staticmethod
    def load_labels(path: Path) -> torch.Tensor:
        with open(path, "rb") as f:
            # Read size (optional, based on C++ code)
            size = np.fromfile(f, dtype=np.uint64, count=1)[0]
            
            # Read the actual data as uint32
            data = np.fromfile(f, dtype=np.uint32, count=size)
        
        return torch.from_numpy(data.astype(np.int32))
    
    @staticmethod
    def zero_mean(point, normalize: bool):
        mean = torch.mean(point, dim=0)
        point -= mean.unsqueeze(0)
        ''' without norm to 1  '''
        if normalize:
            # find maximum distance for each n -> [n]
            furthest_distance = point.pow(2).sum(1).sqrt().max()
            point /= furthest_distance
        return point
    
        
    def load_pointcloud(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                   torch.Tensor, torch.Tensor, torch.Tensor]:
        
        pcl = []
        rgb = []
        descriptors = []
        normals = []
        meshes = []
        for file in path.iterdir():
            if file.suffix == ".bin":
                labels = self.load_labels(file)
                continue
            if file.suffix != ".ply":
                continue
            mesh = o3d.io.read_triangle_mesh(str(file))
            meshes.append(mesh)
            if self.use_normal:
                if not mesh.has_vertex_normals():
                    mesh.compute_vertex_normals()
                if not mesh.has_triangle_normals():
                    mesh.compute_triangle_normals()
            point_cloud = mesh.sample_points_poisson_disk(number_of_points=self.mconfig.node_feature_dim)
            pcl.append(self.norm_tensor(torch.from_numpy(np.asarray(point_cloud.points))))
            descriptors.append(util_data.gen_descriptor_pts(torch.from_numpy(np.asarray(point_cloud.points))))
            rgb.append(torch.from_numpy(np.asarray(point_cloud.colors)))
            if self.use_normal:
                normals.append(torch.tensor(point_cloud.normals))
            else:
                normals.append(torch.zeros_like(pcl[-1]))
                
        num_samples = 10 * self.mconfig.num_points_union
        rel_points = []
        for i in range(len(meshes)):
            for j in range(len(meshes)):
                if i == j:
                    continue
                bbox_i = meshes[i].get_axis_aligned_bounding_box()
                bbox_j = meshes[j].get_axis_aligned_bounding_box()
                min_box = np.minimum(bbox_i.min_bound, bbox_j.min_bound)
                max_box = np.maximum(bbox_i.max_bound, bbox_j.max_bound)
                
                points_i = np.asarray(meshes[i].sample_points_poisson_disk(number_of_points=num_samples).points)
                points_j = np.asarray(meshes[j].sample_points_poisson_disk(number_of_points=num_samples).points)
                all_points = np.concatenate([points_i, points_j], axis=0)
                filter_mask = (all_points[:, 0] > min_box[0]) * (all_points[:, 0] < max_box[0]) \
                            * (all_points[:, 1] > min_box[1]) * (all_points[:, 1] < max_box[1]) \
                            * (all_points[:, 2] > min_box[2]) * (all_points[:, 2] < max_box[2])
                
                instance_i = np.ones((num_samples, 1))
                instance_j = np.ones((num_samples, 1)) * 2
                all_instances = np.concatenate([instance_i, instance_j], axis=0)
                points4d = np.concatenate([all_points, all_instances], axis=1)
                
                pointset = points4d[np.where(filter_mask > 0)[0], :]
                choice = np.random.choice(len(pointset), self.mconfig.num_points_union, replace=True)
                pointset = pointset[choice, :]
                pointset = torch.from_numpy(pointset.astype(np.float32))

                # save_to_ply(pointset[:,:3],'./tmp_rel_{}.ply'.format(e))

                pointset[:, :3] = self.zero_mean(pointset[:, :3], False)
                rel_points.append(pointset)
        if not self.for_eval:
            try:
                rel_points = torch.stack(rel_points, 0)
            except:
                rel_points = torch.zeros([0, self.mconfig.num_points_union, 4])
        else:
            if len(rel_points) == 0:
                # print('len(edge_indices)',len(edge_indices))
                # sometimes tere will have no edge because of only 1 ndoe exist.
                # this is due to the label mapping/filtering process in the data generation
                rel_points = torch.zeros([0, self.mconfig.num_points_union, 4])
            else:
                rel_points = torch.stack(rel_points, 0)
        rel_points = rel_points.permute(0, 2, 1)
    
        descriptors = torch.stack(descriptors, dim=0).float()
        pcl = torch.stack(pcl, dim=0).float()
        rgb = torch.stack(rgb, dim=0).float()
        normals = torch.stack(normals, dim=0).float()
        
        return pcl, rgb, normals, descriptors, labels, rel_points
        
        
    def __getitem__(self, index):
        
        output = HeteroData()
        output['scan_id'] = self.scans[index]
        
        points, rgb, normals, descriptors, _, rel_points = self.load_pointcloud(self.path / self.scans[index])
        num_instances = points.shape[0]
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
        if self.mconfig.rel_data_type == 'points':
            output['edge'].pts = rel_points
                
        return output
    
    def __len__(self):
        return self.size
    
    @staticmethod
    def norm_tensor(points):
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
        