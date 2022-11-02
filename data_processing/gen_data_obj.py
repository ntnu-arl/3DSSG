if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
import os,json, argparse
# import open3d as o3d
import numpy as np
from pathlib import Path
from tqdm import tqdm
import trimesh
from ssg.utils import util_label, util_3rscan, util_data
from ssg import define
from codeLib.utils.util import set_random_seed, read_txt_to_list
import h5py,ast
import copy
import logging

def Parser(add_help=True):
    parser = argparse.ArgumentParser(description='Process some integers.', formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                     add_help=add_help)
    parser.add_argument('--type', type=str, default='train', choices=['train', 'test', 'validation'], help="allow multiple rel pred outputs per pair",required=False)
    parser.add_argument('--label_type', type=str,default='ScanNet20', 
                        choices=['3RScan', '3RScan160', 'NYU40', 'Eigen13', 'RIO27', 'RIO7','ScanNet20'], help='label',required=False)
    parser.add_argument('--pth_out', type=str,default='../data/tmp', help='pth to output directory',required=True)
    parser.add_argument('--relation', type=str,default='relationships', choices=['relationships_extended', 'relationships'])
    parser.add_argument('--target_scan', type=str, default='', help='path to a txt file that contains a list of scan ids that you want to use.')
    parser.add_argument('--scan_name', type=str, default='2dssg_orbslam3', 
                        help='what is the name of the output filename of the ply generated by your segmentation method.')
    
    # options
    parser.add_argument('--debug', type=int, default=0, help='debug',required=False)
    parser.add_argument('--mapping',type=int,default=1,
                        help='map label from 3RScan to label_type. otherwise filter out labels outside label_type.')
    parser.add_argument('--v2', type=int,default=1,
                        help='v2 version')
    parser.add_argument('--inherit', type=int,default=1,help='inherit relationships from the 3RScan.')
    parser.add_argument('--verbose', type=bool, default=False, help='verbal',required=False)
    # parser.add_argument('--scale', type=float,default=1,help='scaling input point cloud.')
    parser.add_argument('--bbox_min_size', type=int, default=100, 
                        help='what is the name of the output filename of the ply generated by your segmentation method.')
    parser.add_argument('--min_entity_num', type=int, default=2, 
                        help='A scene must hhave at least this number of entities.')
    
    # neighbor search parameters
    # parser.add_argument('--search_method', type=str, choices=['BBOX','KNN'],default='BBOX',help='How to split the scene.')
    # parser.add_argument('--radius_receptive', type=float,default=0.5,help='The receptive field of each seed.')
    
    # split parameters
    # parser.add_argument('--split', type=int,default=0,help='Split scene into groups.')
    # parser.add_argument('--radius_seed', type=float,default=1,help='The minimum distance between two seeds.')
    # parser.add_argument('--min_segs', type=int,default=5,help='Minimum segments for each segGroup')
    # parser.add_argument('--split_method', type=str, choices=['BBOX','KNN'],default='BBOX',help='How to split the scene.')
    
    # Correspondence Parameters
    # parser.add_argument('--max_dist', type=float,default=0.1,help='maximum distance to find corresopndence.')
    parser.add_argument('--min_3D_bbox_size', type=int,default=0.2*0.2*0.2,help='minimum bounding box region (m^3).')
    # parser.add_argument('--corr_thres', type=float,default=0.5,help='How the percentage of the points to the same target segment must exceeds this value.')
    parser.add_argument('--occ_thres', type=float,default=0.5,help='the fraction of GT labels in a segment should exceed this')
    
    # constant
    parser.add_argument('--segment_type', type=str,default='ORBSLAM3')
    return parser

debug = True
debug = False

def process(pth_3RScan, scan_id, target_relationships:list,
            gt_relationships:dict=None) -> list:
    ''' load instance mapping'''
    _, label_name_mapping, _ = util_label.getLabelMapping(args.label_type)
    segseg_file_name = 'semseg.v2.json' if args.v2 else 'semseg.json'
    pth_semseg_file = os.path.join(pth_3RScan, scan_id, segseg_file_name)
    instance2labelName = util_3rscan.load_semseg(pth_semseg_file, label_name_mapping,args.mapping)
    
    ''' load graph file '''
    pth_pd = os.path.join(pth_3RScan,scan_id,args.scan_name+'.json')
    with open(pth_pd,'r') as f:
        data = json.load(f)[scan_id]
    graph = util_data.load_graph(data,box_filter_size=[int(args.bbox_min_size)])
    nodes = graph['nodes']
    # keyframes = graph['kfs']
    
    '''check at node has valid points'''
    # Check file
    pth_ply = os.path.join(pth_3RScan,scan_id,args.scan_name+'.ply')
    if not os.path.isfile(pth_ply):
        logger_py.info('skip {} due to no ply file exists'.format(scan_id))
        return [], []
        # raise RuntimeError('cannot find file {}'.format(pth_ply))
    # Load
    cloud_pd = trimesh.load(pth_ply, process=False)
    segments_pd = cloud_pd.metadata['ply_raw']['vertex']['data']['label'].flatten()
    # Get IDs
    segment_ids_pts = np.unique(segments_pd) 
    segment_ids_pts = segment_ids_pts[segment_ids_pts!=0]
    
    ''' get number of nodes '''
    segment_ids = [k for k in nodes.keys()]
    if 0 in segment_ids: segment_ids.remove(0) # ignore none
    if args.verbose: print('filtering input segments.. (ori num of segments:',len(segment_ids),')')
    segments_pd_filtered=list()
    map_segment_pd_2_gt = dict() # map segment_pd to segment_gt
    gt_segments_2_pd_segments = dict() # how many segment_pd corresponding to this segment_gt
    segs_neighbors=dict()
    for seg_id in segment_ids:
        node = nodes[seg_id]
        if node.kfs is None or len(node.kfs) == 0:
            print('warning. each node should have at least 1 kf')
        
        if node.size() <= args.min_3D_bbox_size :
            # name = instance2labelName.get(seg_id,'unknown')
            if debug: print('node',seg_id,'too small (', node.size(),'<',args.min_3D_bbox_size,')')
            continue
        
        # Check at least has a valid pts
        pts = segment_ids_pts[np.where(segment_ids_pts==seg_id)]
        if len(pts) == 0:
            continue
        
        '''find GT instance'''
        # get maximum
        max_v=0
        max_k=0
        for k,v in node.gtInstance.items():
            if v>max_v:
                max_v=v
                max_k=int(k)
        if max_v < args.occ_thres:
            if debug: print('node',seg_id,'has too small overlappign to GT instance', max_v,'<',args.occ_thres)
            continue
        
        '''skip nonknown'''
        if instance2labelName[max_k] == '-' or instance2labelName[max_k] =='none':
            if debug: print('node',seg_id,'has unknown GT instance',max_k)
            continue
        
        '''  '''
        map_segment_pd_2_gt[int(seg_id)] = int(max_k)
        
        if max_k not in gt_segments_2_pd_segments:
            gt_segments_2_pd_segments[max_k] = list()
        gt_segments_2_pd_segments[max_k].append(seg_id)        
        
        segs_neighbors[int(seg_id)] = node.neighbors

        segments_pd_filtered.append(seg_id)
    segment_ids = segments_pd_filtered
    if debug: 
        print('there are',len(segment_ids), 'segemnts:\n', segment_ids)
        print('sid iid label')
        for sid in segment_ids:
            print(sid,':',map_segment_pd_2_gt[sid],instance2labelName[map_segment_pd_2_gt[sid]])
            
    if len(segment_ids) < args.min_entity_num:
        if debug: print('num of entities ({}) smaller than {}'.format(len(segment_ids),args.min_entity_num))
        return [],{}
    
    '''process'''
    objs_obbinfo=dict()
    with open(pth_pd,'r') as f:
        data = json.load(f)[scan_id]
    for nid, node in data['nodes'].items():
        # if nid not in segment_ids: continue
        obj_obbinfo = objs_obbinfo[int(nid)] = dict()
        obj_obbinfo['center'] = copy.deepcopy(node['center'])
        obj_obbinfo['dimension'] = copy.deepcopy(node['dimension'])
        obj_obbinfo['normAxes'] = copy.deepcopy( np.array(node['rotation']).reshape(3,3).transpose().tolist() )
    del data
    
    relationships = gen_relationship(scan_id, 0,
                                     map_segment_pd_2_gt, 
                                     instance2labelName, 
                                     gt_segments_2_pd_segments,
                                     gt_relationships,
                                     target_relationships)
    
    list_relationships = list()
    if len(relationships["objects"]) != 0 and len(relationships['relationships']) != 0:
        list_relationships.append(relationships)
                
    for relationships in list_relationships:
        for oid in relationships['objects'].keys():
            relationships['objects'][oid] = {**objs_obbinfo[oid], **relationships['objects'][oid]}
    return list_relationships, segs_neighbors


def gen_relationship(scan_id:str,split:int, 
                     map_segment_pd_2_gt:dict,
                     instance2labelName:dict,
                     gt_segments_2_pd_segments:dict,
                     gt_relationships,
                     target_relationships,
                     target_segments:list=None) -> dict:
    '''' Save as relationship_*.json '''
    relationships = dict() #relationships_new["scans"].append(s)
    relationships["scan"] = scan_id
    relationships["split"] = split
    
    objects = dict()
    for seg, segment_gt in map_segment_pd_2_gt.items():
        if target_segments is not None:
            if seg not in target_segments: continue
        name = instance2labelName[segment_gt]
        assert(name != '-' and name != 'none')
        objects[int(seg)] = dict()
        objects[int(seg)]['label'] = name
        objects[int(seg)]['instance_id'] = segment_gt
    relationships["objects"] = objects
    
    
    split_relationships = list()
    ''' Inherit relationships from ground truth segments '''
    if gt_relationships is not None:
        relationships_names = read_txt_to_list(os.path.join(define.FILE_PATH, args.relation + ".txt"))

        for rel in gt_relationships:
            id_src = rel[0]
            id_tar = rel[1]
            num = rel[2]
            name = rel[3]
            idx_in_txt = relationships_names.index(name)
            assert(num==idx_in_txt)
            if name not in target_relationships: 
                continue
            if id_src == id_tar:
                if debug:print('ignore relationship (',name,') between',id_src,'and',id_tar,'that has the same source and target')
                continue
            idx_in_txt_new = target_relationships.index(name)
            
            if id_src in gt_segments_2_pd_segments and id_tar in gt_segments_2_pd_segments:
                segments_src = gt_segments_2_pd_segments[id_src]
                segments_tar = gt_segments_2_pd_segments[id_tar]                
                for segment_src in segments_src:
                    if segment_src not in objects:
                        if debug:print('filter',name,'segment_src', instance2labelName[id_src],' is not in objects')
                        continue
                    for segment_tar in segments_tar:        
                        if segment_tar not in objects:
                            if debug:print('filter',name,'segment_tar', instance2labelName[id_tar], ' is not in objects')
                            continue
                        if target_segments is not None:
                            ''' skip if they not in the target_segments'''
                            if segment_src not in target_segments: continue
                            if segment_tar not in target_segments: continue
                        # if segment_tar == segments_src:continue
                        ''' check if they are neighbors '''
                        split_relationships.append([ int(segment_src), int(segment_tar), idx_in_txt_new, name ])
                        if debug:print('inherit', instance2labelName[id_src],'(',id_src,')',name, instance2labelName[id_tar],'(',id_tar,')')
            # else:
            #     if debug:
            #         if id_src in gt_segments_2_pd_segments:
            #             print('filter', instance2labelName[id_src],name, instance2labelName[id_tar],'id_src', id_src, 'is not in the gt_segments_2_pd_segments')
            #         if id_tar in gt_segments_2_pd_segments:
            #             print('filter', instance2labelName[id_src],name, instance2labelName[id_tar],'id_tar', id_tar, 'is not in the gt_segments_2_pd_segments')
    
    ''' Build "same part" relationship '''
    idx_in_txt_new = target_relationships.index(define.NAME_SAME_PART)
    for _, groups in gt_segments_2_pd_segments.items():
        if target_segments is not None:
            filtered_groups = list()
            for g in groups:
                if g in target_segments:
                    filtered_groups.append(g)
            groups = filtered_groups
        if len(groups) <= 1: continue
                    
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                split_relationships.append([int(groups[i]),int(groups[j]), idx_in_txt_new, define.NAME_SAME_PART])
                split_relationships.append([int(groups[j]),int(groups[i]), idx_in_txt_new, define.NAME_SAME_PART])
    
    relationships["relationships"] = split_relationships
    return relationships
    
if __name__ == '__main__':
    args = Parser().parse_args()
    Path(args.pth_out).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.pth_out,'gen_data_'+args.type+'.log'), level=logging.DEBUG)
    logger_py = logging.getLogger(__name__)
    
    debug |= args.debug>0
    args.verbose |= debug
    # if args.search_method == 'BBOX':
    #     search_method = SAMPLE_METHODS.BBOX
    # elif args.search_method == 'KNN':
    #     search_method = SAMPLE_METHODS.RADIUS
    set_random_seed(2020)
    label_names, _, _ = util_label.getLabelMapping(args.label_type)
    classes_json = list()
    for key,value in label_names.items():
        if value == '-':continue
        classes_json.append(value)
        
    ''' Read Scan and their type=['train', 'test', 'validation'] '''
    scan2type = {}
    with open(define.Scan3RJson_PATH, "r") as read_file:
        data = json.load(read_file)
        for scene in data:
            scan2type[scene["reference"]] = scene["type"]
            for scan in scene["scans"]:
                scan2type[scan["reference"]] = scene["type"]
    
    '''read relationships'''
    target_relationships=list()
    if args.inherit:
        # target_relationships += ['supported by', 'attached to','standing on', 'lying on','hanging on','connected to',
                                # 'leaning against','part of','build in','standing in','lying in','hanging in']
        target_relationships += ['supported by', 'attached to','standing on','hanging on','connected to','part of','build in']
    target_relationships.append(define.NAME_SAME_PART)
    
    target_scan=[]
    if args.target_scan != '':
        target_scan = read_txt_to_list(args.target_scan)
        
    valid_scans=list()
    relationships_new = dict()
    relationships_new["scans"] = list()
    relationships_new['neighbors'] = dict()
    counter= 0
    with open(os.path.join(define.FILE_PATH + args.relation + ".json"), "r") as read_file:
        data = json.load(read_file)
        filtered_data = list()
        
        for s in data["scans"]:
            scan_id = s["scan"]
            if len(target_scan) ==0:
                if scan2type[scan_id] != args.type: 
                    if args.verbose:
                        print('skip',scan_id,'not validation type')
                    continue
            else:
                if scan_id not in target_scan: continue
            
            filtered_data.append(s)

        for s in tqdm(filtered_data):
            scan_id = s["scan"]
            gt_relationships = s["relationships"]
            logger_py.info('processing scene {}'.format(scan_id))
            valid_scans.append(scan_id)
            
            relationships, segs_neighbors = process(os.path.join('data','3RScan','data','3RScan'), scan_id, target_relationships,
                                    gt_relationships = gt_relationships)
            if len(relationships) == 0:
                logger_py.info('skip {} due to not enough objs and relationships'.format(scan_id))
                continue
            else:
                if debug:  print('no skip', scan_id)
            
            relationships_new["scans"] += relationships
            relationships_new['neighbors'][scan_id] = segs_neighbors
            
            if debug:
                break
            
    '''Save'''
    pth_args = os.path.join(args.pth_out,'args.json')
    with open(pth_args, 'w') as f:
            tmp = vars(args)
            json.dump(tmp, f, indent=2)
            
    pth_classes = os.path.join(args.pth_out, 'classes.txt')
    with open(pth_classes,'w') as f:
        for name in classes_json:
            if name == '-': continue
            f.write('{}\n'.format(name))
    pth_relation = os.path.join(args.pth_out, 'relationships.txt')
    with open(pth_relation,'w') as f:
        for name in target_relationships:
            f.write('{}\n'.format(name))
    pth_split = os.path.join(args.pth_out, args.type+'_scans.txt')
    with open(pth_split,'w') as f:
        for name in valid_scans:
            f.write('{}\n'.format(name))
    # '''Save to json'''
    # pth_relationships_json = os.path.join(args.pth_out, "relationships_" + args.type + ".json")
    # with open(pth_relationships_json, 'w') as f:
    #     json.dump(relationships_new, f)
        
    '''Save to h5'''
    pth_relationships_json = os.path.join(args.pth_out, "relationships_" + args.type + ".h5")
    h5f = h5py.File(pth_relationships_json, 'w')
    # reorganize scans from list to dict
    scans = dict()
    for s in relationships_new['scans']:
        scans[s['scan']] = s
    all_neighbors = relationships_new['neighbors']
    for scan_id in scans.keys():
        scan_data = scans[scan_id]
        neighbors = all_neighbors[scan_id]
        objects = scan_data['objects']
        
        d_scan = dict()
        d_nodes = d_scan['nodes'] = dict()
        
        ## Nodes
        for idx, data in enumerate(objects.items()):
            oid, obj_info = data
            ascii_nn = [str(n).encode("ascii", "ignore") for n in neighbors[oid]]
            d_nodes[oid] = dict()
            d_nodes[oid] = obj_info
            d_nodes[oid]['neighbors'] = ascii_nn
        
        ## Relationships
        str_relationships = list() 
        for rel in scan_data['relationships']:
            str_relationships.append([str(s) for s in rel])
        d_scan['relationships']= str_relationships
        
        s_scan = str(d_scan)
        h5_scan = h5f.create_dataset(scan_id,data=np.array([s_scan],dtype='S'),compression='gzip')
        # test decode 
        tmp = h5_scan[0].decode()
        assert isinstance(ast.literal_eval(tmp),dict)
        
        # ast.literal_eval(h5_scan)
    h5f.close()
    
