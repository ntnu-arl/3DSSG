import os
import logging
import codeLib
import ssg
import ssg.config as config
from ssg.checkpoints import CheckpointIO
import cProfile
import matplotlib
import torch_geometric
import torch
import numpy as np
import json
import tqdm

# disable GUI
matplotlib.pyplot.switch_backend('agg')
# change log setting
matplotlib.pyplot.set_loglevel("CRITICAL")
logging.getLogger('PIL').setLevel('CRITICAL')
logging.getLogger('trimesh').setLevel('CRITICAL')
logging.getLogger("h5py").setLevel(logging.INFO)
logger_py = logging.getLogger(__name__)


def main():
    cfg = ssg.Parse()

    # Shorthands
    out_dir = os.path.join(cfg['training']['out_dir'])
    # set random seed
    codeLib.utils.util.set_random_seed(cfg.SEED)

    # Output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Log
    logging.basicConfig(filename=os.path.join(
        out_dir, 'log'), level=cfg.log_level)
    logger_py.setLevel(cfg.log_level)
    
    dataset_test = config.get_dataset(cfg, 'test')
    dataset_test.save_map = True
    dataset_test.save_map_path = out_dir
    val_loader = torch_geometric.loader.DataLoader(
                dataset_test, batch_size=1, num_workers=cfg['eval']['data_workers'],
                shuffle=False, drop_last=False,
                pin_memory=False,
            )
    logger_py.info('test loader')
    logger = config.get_logger(cfg)
    if logger is not None:
        logger, _ = logger
    relationNames = dataset_test.relationNames
    num_obj_cls = len(dataset_test.classNames)
    num_rel_cls = len(
        dataset_test.relationNames) if relationNames is not None else 0

    model = config.get_model(
    cfg, num_obj_cls=num_obj_cls, num_rel_cls=num_rel_cls)

    checkpoint_io = CheckpointIO(cfg['training']['model_dir'], model=model)
    
    load_dict = checkpoint_io.load('model_best.pt', device=cfg.DEVICE)
    it = load_dict.get('it', -1)

    #
    logger_py.info('start evaluation')
    pr = cProfile.Profile()
    pr.enable()
    model.eval()
    for data in tqdm.tqdm(val_loader):
        with torch.inference_mode():
            data = data.to(cfg.DEVICE)
            node_cls, edge_cls = model(data)
            node_cls = node_cls.cpu()
            edge_cls = edge_cls.cpu()
            node_cls_pred_probs= torch.softmax(node_cls, dim=1)
            node_cls_pred = torch.max(node_cls_pred_probs, 1)[1]
            edge_cls_pred_probs = torch.softmax(edge_cls, dim=1)
            edge_cls_pred = torch.max(edge_cls_pred_probs, 1)[1]
            node_name_pred = np.asarray(dataset_test.classNames)[node_cls_pred.numpy()].tolist()
            edge_name_pred = dict()
            i = 0
            for edge in data['node', 'to', 'node'].edge_index.t():
                if edge[0].item() not in edge_name_pred:
                    edge_name_pred[edge[0].item()] = {edge[1].item() : relationNames[edge_cls_pred[i].item()]}
                else:
                    edge_name_pred[edge[0].item()][edge[1].item()] = relationNames[edge_cls_pred[i].item()]
                i += 1
                
            # edge_name_pred = {i:{data['node', 'to', 'node'].edge_index[1][i * (len(node_name_pred) - 1) + j].item() : relationNames[edge_cls_pred[i * (len(node_name_pred) - 1) + j].item()] for j in range(len(node_name_pred) - 1)} for i in range(len(node_name_pred))}
            results = dict()
            results[data['scan_id'][0]] = {'nodes': node_name_pred, 
                                           'edges': edge_name_pred, 
                                           'node_probs': node_cls_pred_probs.numpy().tolist(),
                                           'edge_probs': edge_cls_pred_probs.numpy().tolist()}
            with open(os.path.join(out_dir, f'{data["scan_id"][0]}.json'), 'w') as f:
                json.dump(results, f, indent=4)
            
    pr.disable()
    logger_py.info('save time profile to {}'.format(
        os.path.join(out_dir, 'tp_eval.dmp')))
    pr.dump_stats(os.path.join(out_dir, 'tp_eval.dmp'))
    
    
if __name__ == '__main__':
    main()
        
    
    
       