import copy
from tqdm import tqdm
from collections import defaultdict
from codeLib.models import BaseTrainer
from codeLib.common import check_weights, convert_torch_to_scalar
import torch
from ssg.utils.util_eva import EvalSceneGraph
import time
import logging
import codeLib.utils.moving_average as moving_average 
import ssg
import codeLib.utils.string_numpy as snp
from ssg import define
logger_py = logging.getLogger(__name__)

class Trainer_IMP(BaseTrainer):
    def __init__(self, cfg, model, node_cls_names:list, edge_cls_names:list,
                 device=None,  **kwargs):
        super().__init__(device)
        logger_py.setLevel(cfg.log_level)
        self.cfg = cfg
        self.model = model#.to(self._device)
        self.w_node_cls = kwargs.get('w_node_cls',None)
        self.w_edge_cls = kwargs.get('w_edge_cls',None)
        self.node_cls_names = node_cls_names#kwargs['node_cls_names']
        self.edge_cls_names = edge_cls_names#kwargs['edge_cls_names']
        
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = ssg.config.get_optimizer(cfg, trainable_params)
        
        if self.w_node_cls is not None: 
            logger_py.info('train with weighted node class.')
            self.w_node_cls = self.w_node_cls.to(self._device)
        if self.w_edge_cls is not None: 
            logger_py.info('train with weighted node class.')
            self.w_edge_cls= self.w_edge_cls.to(self._device)
                
        self.eva_tool = EvalSceneGraph(self.node_cls_names, self.edge_cls_names,multi_rel_prediction=self.cfg.model.multi_rel,k=0,none_name=define.NAME_NONE) # do not calculate topK in training mode        
        self.loss_node_cls = torch.nn.CrossEntropyLoss(weight=self.w_node_cls)
        if self.cfg.model.multi_rel:
            self.loss_rel_cls = torch.nn.BCEWithLogitsLoss(pos_weight=self.w_edge_cls)
        else:
            self.loss_rel_cls = torch.nn.CrossEntropyLoss(weight=self.w_edge_cls)

    def zero_metrics(self): 
        self.eva_tool.reset()
        
    def evaluate(self, val_loader, topk):
        it_dataset = val_loader.__iter__()
        eval_tool = EvalSceneGraph(self.node_cls_names, self.edge_cls_names,multi_rel_prediction=self.cfg.model.multi_rel,k=topk,save_prediction=True,
                                   none_name=define.NAME_NONE) 
        eval_list = defaultdict(moving_average.MA)

        time.sleep(2)# Prevent possible deadlock during epoch transition
        for data in tqdm(it_dataset,leave=False):
            eval_step_dict = self.eval_step(data,eval_tool=eval_tool)
            
            for k, v in eval_step_dict.items():
                eval_list[k].update(v)
            # break
        eval_dict = dict()
        obj_, edge_ = eval_tool.get_mean_metrics()
        for k,v in obj_.items():
            # print(k)
            eval_dict[k+'_node_cls'] = v
        for k,v in edge_.items():
            # print(k)
            eval_dict[k+'_edge_cls'] = v
        
        for k, v in eval_list.items():
            eval_dict[k] = v.avg
        
        vis = self.visualize(eval_tool=eval_tool)
        eval_dict['visualization'] = vis
        del it_dataset
        return eval_dict, eval_tool
    
    def evaluate_inst(self,dataset_seg,dataset_inst,topk):
        ignore_missing=self.cfg.eval.ignore_missing
        '''add a none class for missing instances'''
        node_cls_names = copy.copy(self.node_cls_names)
        edge_cls_names = copy.copy(self.edge_cls_names)
        if define.NAME_NONE not in self.node_cls_names:
            node_cls_names.append(define.NAME_NONE)
        if define.NAME_NONE not in self.edge_cls_names:
            edge_cls_names.append(define.NAME_NONE)
        # remove same part
        # samepart_idx_edge_cls = self.edge_cls_names.index(define.NAME_SAME_PART)
        if define.NAME_SAME_PART in edge_cls_names:
            edge_cls_names.remove(define.NAME_SAME_PART)
        
        noneidx_node_cls = node_cls_names.index(define.NAME_NONE)
        noneidx_edge_cls = edge_cls_names.index(define.NAME_NONE)
        
        seg_valid_edge_cls_indices = []
        for idx in range(len(self.edge_cls_names)):
            name = self.edge_cls_names[idx]
            if name == define.NAME_SAME_PART: continue
            seg_valid_edge_cls_indices.append(idx)
        
        
        eval_tool = EvalSceneGraph(node_cls_names, edge_cls_names,multi_rel_prediction=self.cfg.model.multi_rel,k=topk,save_prediction=True,
                                   none_name=define.NAME_NONE) 
        eval_list = defaultdict(moving_average.MA)
        
        '''check'''
        #length
        print('len(dataset_seg), len(dataset_inst):',len(dataset_seg), len(dataset_inst))
        print('ignore missing',ignore_missing)
        #classes
        
        
        ''' get scan_idx mapping '''
        scanid2idx_seg = dict()
        for index in range(len(dataset_seg)):
            scan_id = snp.unpack(dataset_seg.scans,index)# self.scans[idx]
            scanid2idx_seg[scan_id] = index
            
        scanid2idx_inst = dict()
        for index in range(len(dataset_inst)):
            scan_id = snp.unpack(dataset_inst.scans,index)# self.scans[idx]
            scanid2idx_inst[scan_id] = index
            
        
        # '''build main seg loader'''
        # seg_dataloader = torch.utils.data.DataLoader(
        #     dataset_seg, batch_size=1, num_workers=0,
        #     shuffle=False, drop_last=False,
        #     pin_memory=True,
        #     collate_fn=graph_collate,
        # )
        
        '''start eval'''
        
        self.model.eval()
        for index in tqdm(range(len(dataset_inst))):
        # for data_inst in seg_dataloader:
            data_inst = dataset_inst.__getitem__(index)                
            scan_id_inst = data_inst['scan_id']
            
            if scan_id_inst not in scanid2idx_seg:
                #TODO: what should we do if missing scans?
                continue
            
            index_seg = scanid2idx_seg[scan_id_inst]
            data_seg  = dataset_seg.__getitem__(index_seg)
            
            assert data_seg['scan_id'] == data_inst['scan_id']
            
            
            '''process seg'''
            eval_dict={}
            with torch.no_grad():
                logs = {}
        
                # Process data dictionary
                data = self.process_data_dict(data_seg)
                data_inst = self.process_data_dict(data_inst)
                
                # Shortcuts
                scan_id = data['scan_id']
                gt_cls = data['gt_cls']
                gt_rel = data['gt_rel']
                instance2mask = data['instance2mask']
                node_edges_ori = data['node_edges']
                data['node_edges'] = data['node_edges'].t().contiguous()
                
                inst_instance2mask = data_inst['instance2mask']
                inst_gt_cls = data_inst['gt_cls']
                inst_gt_rel = data_inst['gt_rel']
                data_inst['node_edges'] = data_inst['node_edges'].t().contiguous()
                
                # check input valid
                if node_edges_ori.ndim==1: continue
                
                ''' make forward pass through the network '''
                node_cls, edge_cls = self.model(**data)
                
                '''merge prediction from seg to instance'''
                seg2inst = data['seg2inst']
                inst2masks = defaultdict(list)
                for seg,inst in seg2inst.items():
                    mask = instance2mask[seg]
                    inst2masks[inst].append(mask)
                
                # merge predictions on the instances
                merged_instance2idx = dict()
                merged_node_cls = torch.zeros(len(inst_instance2mask),len(node_cls_names))
                merged_node_cls_gt = torch.zeros(len(inst_instance2mask),dtype=torch.long)
                counter=0
                for idx, (inst, mask )in enumerate(inst_instance2mask.items()):                    
                    # merge predictions
                    if not ignore_missing:
                        merged_instance2idx[inst] = idx
                        merged_node_cls_gt[idx] = inst_gt_cls[mask]
                        if inst in inst2masks:
                            '''merge nodes'''
                            predictions = node_cls[inst2masks[inst]]
                            
                            # averaging the probability
                            node_cls_pred = torch.softmax(predictions, dim=1).mean(dim=0)                        
                            merged_node_cls[idx][:node_cls_pred.shape[0]] = node_cls_pred
                            
                            # node_cls_pred = torch.max(node_cls_pred,1)[1]
                            
                            '''merge edge'''
                        else:
                            #TODO: handle missing inst
                            merged_node_cls[idx][noneidx_node_cls] = 1.0
                            # merged_node_cls_gt[idx]=noneidx_node_cls
                            
                            
                            pass
                    else:
                        if inst not in inst2masks:
                            continue
                        merged_instance2idx[inst] = counter
                        predictions = node_cls[inst2masks[inst]]
                        node_cls_pred = torch.softmax(predictions, dim=1).mean(dim=0)                        
                        merged_node_cls[counter][:node_cls_pred.shape[0]] = node_cls_pred
                        merged_node_cls_gt[counter] = inst_gt_cls[mask]
                        counter+=1
                if ignore_missing:
                    merged_node_cls = merged_node_cls[:counter]
                    merged_node_cls_gt = merged_node_cls_gt[:counter]
                '''merge '''
                idx2seg= merge_batch_seg2idx(instance2mask) 
                inst_idx2seg=merge_batch_seg2idx(inst_instance2mask)
                
                # build search list for GT edge pairs
                inst_gt_pairs = set()
                inst_gt_rel_dict = dict()
                for idx in range(len(inst_gt_rel)):
                    src_idx, tgt_idx = data_inst['node_edges'][0,idx].item(),data_inst['node_edges'][1,idx].item()
                    src_inst_idx, tgt_inst_idx = inst_idx2seg[src_idx], inst_idx2seg[tgt_idx]
                    inst_gt_pairs.add((src_inst_idx, tgt_inst_idx))
                    inst_gt_rel_dict[(src_inst_idx, tgt_inst_idx)] = inst_gt_rel[idx]
               
                merged_edge_cls_dict = defaultdict(list)
                for idx in range(len(gt_rel)):
                    src_idx, tgt_idx = data['node_edges'][0,idx].item(), data['node_edges'][1,idx].item()
                    relname = self.edge_cls_names[gt_rel[idx].item()]
                    
                    src_seg_idx = idx2seg[src_idx]
                    src_inst_idx = seg2inst[src_seg_idx]
                    
                    tgt_seg_idx = idx2seg[tgt_idx]
                    tgt_inst_idx = seg2inst[tgt_seg_idx]
                    pair = (src_inst_idx,tgt_inst_idx)
                    if  pair in inst_gt_pairs:
                        merged_edge_cls_dict[pair].append( edge_cls[idx] )
                    else:
                        # print('cannot find seg:{}(inst:{}) to seg:{}(inst:{}) with relationship:{}.'.format(src_seg_idx,src_inst_idx,tgt_seg_idx,tgt_inst_idx,relname))
                        pass
                # check missing rels
                # for pair in inst_gt_rel_dict:
                #     if pair not in merged_edge_cls_dict:
                #         relname = self.edge_cls_names[inst_gt_rel_dict[pair]]
                #         src_name = self.node_cls_names[inst_gt_cls[inst_instance2mask[pair[0]]]]
                #         tgt_name = self.node_cls_names[inst_gt_cls[inst_instance2mask[pair[1]]]]
                #         print('missing inst:{}({}) to inst:{}({}) with relationship:{}'.format(pair[0],src_name,pair[1],tgt_name,relname))
                        
                '''merge predictions'''
                merged_edge_cls = torch.zeros(len(inst_gt_rel),len(edge_cls_names))
                merged_edge_cls_gt = torch.zeros(len(inst_gt_rel),dtype=torch.long)
                merged_node_edges = list()
                counter=0
                for idx, (pair,inst_edge_cls) in enumerate(inst_gt_rel_dict.items()):
                    merged_edge_cls_gt[counter] = inst_gt_rel_dict[pair]
                    if ignore_missing:
                        if pair[0] not in merged_instance2idx: continue
                        if pair[1] not in merged_instance2idx: continue
                    mask_src = merged_instance2idx[pair[0]]
                    mask_tgt = merged_instance2idx[pair[1]]
                    merged_node_edges.append([mask_src,mask_tgt])
                    
                    if pair in merged_edge_cls_dict:
                        edge_pds = torch.stack(merged_edge_cls_dict[pair])
                        edge_pds = edge_pds[:,seg_valid_edge_cls_indices]
                        # seg_valid_edge_cls_indices
                        
                        #ignore same part
                        
                        edge_pds = torch.softmax(edge_pds,dim=1).mean(0)
                        merged_edge_cls[counter][:edge_pds.shape[0]] = edge_pds
                    else:
                        merged_edge_cls[counter][noneidx_edge_cls] = 1.0
                    counter+=1
                if ignore_missing:
                    merged_edge_cls=merged_edge_cls[:counter]
                merged_node_edges = torch.tensor(merged_node_edges,dtype=torch.long)

            eval_tool.add([scan_id], 
                          merged_node_cls,
                          merged_node_cls_gt, 
                          merged_edge_cls,
                          merged_edge_cls_gt,
                          [merged_instance2idx],
                          merged_node_edges)
        eval_dict = dict()
        obj_, edge_ = eval_tool.get_mean_metrics()
        for k,v in obj_.items():
            # print(k)
            eval_dict[k+'_node_cls'] = v
        for k,v in edge_.items():
            # print(k)
            eval_dict[k+'_edge_cls'] = v
        
        for k, v in eval_list.items():
            eval_dict[k] = v.avg
        
        vis = self.visualize(eval_tool=eval_tool)
        eval_dict['visualization'] = vis
        return eval_dict, eval_tool
    
    def sample(self, dataloader):
        pass
        
    def train_step(self, data, it=None):
        self.model.train()
        self.optimizer.zero_grad()
        logs = self.compute_loss(data, it=it, eval_tool = self.eva_tool)
        if 'loss' not in logs: return logs
        logs['loss'].backward()
        check_weights(self.model.state_dict())
        self.optimizer.step()
        return logs
    
    def eval_step(self,data, eval_tool=None):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        eval_dict={}
        with torch.no_grad():
            eval_dict = self.compute_loss(
                data, eval_mode=True, eval_tool=eval_tool)
        eval_dict = convert_torch_to_scalar(eval_dict)
        # for (k, v) in eval_dict.items():
        #     eval_dict[k] = v.item()
        return eval_dict
    
    def process_data_dict(self, data):
        ''' Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary
        '''
        data =  dict(zip(data.keys(), self.toDevice(*data.values()) ))
        return data
    
    def compute_loss(self,data,eval_mode=False,it=None, eval_tool=None):
        ''' Compute the loss.

        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
        '''
        # Initialize loss dictionary and other values
        logs = {}
        
        # Process data dictionary
        data = self.process_data_dict(data)
        
        # Shortcuts
        scan_id = data['scan_id']
        gt_cls = data['gt_cls']
        gt_rel = data['gt_rel']
        mask2instance = data['mask2instance']
        node_edges_ori = data['node_edges']
        data['node_edges'] = data['node_edges'].t().contiguous()
        
        # check input valid
        if node_edges_ori.ndim==1:
            return {}
        
        # print('')
        # print('gt_rel.sum():',gt_rel.sum())
        
        ''' make forward pass through the network '''
        node_cls, edge_cls = self.model(**data)
        
        ''' calculate loss '''
        logs['loss'] = 0
        
        
        if self.cfg.training.lambda_mode == 'dynamic':
            # calculate loss ratio base on the number of node and edge
            batch_node = node_cls.shape[0]
            batch_edge = edge_cls.shape[0]
            self.cfg.training.lambda_node = 1
            self.cfg.training.lambda_edge = batch_edge / batch_node
            
        
        ''' 1. node class loss'''
        self.calc_node_loss(logs, node_cls, gt_cls, self.w_node_cls)
        
        ''' 2. edge class loss '''
        self.calc_edge_loss(logs, edge_cls, gt_rel, self.w_edge_cls)
        
        '''3. get metrics'''
        metrics = self.model.calculate_metrics(
            node_cls_pred=node_cls,
            node_cls_gt=gt_cls,
            edge_cls_pred=edge_cls,
            edge_cls_gt=gt_rel
        )
        for k,v in metrics.items():
            logs[k]=v
            
        ''' eval tool '''
        if eval_tool is not None:
            node_cls = torch.softmax(node_cls.detach(),dim=1)
            edge_cls = torch.sigmoid(edge_cls.detach())
            eval_tool.add(scan_id, 
                          node_cls,gt_cls, 
                          edge_cls,gt_rel,
                          mask2instance,node_edges_ori)
        return logs
        # return loss if eval_mode else loss['loss']

    def calc_node_loss(self, logs, node_cls_pred, node_cls_gt, weights=None):
        '''
        calculate node loss.
        can include
        classification loss
        attribute loss
        affordance loss
        '''
        # loss_obj = F.nll_loss(node_cls_pred, node_cls_gt, weight = weights)
        loss_obj = self.loss_node_cls(node_cls_pred, node_cls_gt)
        logs['loss'] += self.cfg.training.lambda_node * loss_obj
        logs['loss_obj'] = loss_obj
        
    def calc_edge_loss(self, logs, edge_cls_pred, edge_cls_gt, weights=None):
        if self.cfg.model.multi_rel:
            # batch_mean = torch.sum(edge_cls_gt, dim=(0))
            # zeros = (edge_cls_gt ==0).sum().unsqueeze(0)
            # batch_mean = torch.cat([zeros,batch_mean],dim=0)
            # weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf                
            # weight[torch.where(weight==0)] = weight[0].clone()
            # weight = weight[1:]
            # pass
            loss_rel = self.loss_rel_cls(edge_cls_pred, edge_cls_gt)
        else:
            loss_rel = self.loss_rel_cls(edge_cls_pred, edge_cls_gt)
            # loss_rel = F.binary_cross_entropy(edge_cls_pred, edge_cls_gt, weight=weights)
        # else:
            # loss_rel = self.loss_rel_cls(edge_cls_pred,edge_cls_gt)
            # loss_rel = F.nll_loss(edge_cls_pred, edge_cls_gt, weight = weights)
        logs['loss'] += self.cfg.training.lambda_edge * loss_rel
        logs['loss_rel'] = loss_rel
    # def convert_metrics_to_log(self, metrics, eval_mode=False):
    #     tmp = dict()
    #     mode = 'train_' if eval_mode == False else 'valid_'
    #     for metric_name, dic in metrics.items():
    #         for sub, value in dic.items():
    #             tmp[metric_name+'/'+mode+sub] = value
    #     return tmp
    # def calc_node_metric(self, logs, node_cls_pred, node_cls_gt):
    #     cls_pred = node_cls_pred.detach()
    #     pred_cls = torch.max(cls_pred,1)[1]
    #     acc_cls = (node_cls_gt == pred_cls).sum().item() / node_cls_gt.nelement()
    #     logs['acc_node_cls'] = acc_cls
        
    
        
    def visualize(self,eval_tool=None):
        if eval_tool is None: eval_tool = self.eva_tool
        node_confusion_matrix, edge_confusion_matrix = eval_tool.draw(
            plot_text=False,
            grid=False,
            normalize='log',
            plot = False
            )
        return {
            'node_confusion_matrix': node_confusion_matrix,
            'edge_confusion_matrix': edge_confusion_matrix
        }
    
    def toDevice(self, *args):
        output = list()
        for item in args:
            if isinstance(item,  torch.Tensor):
                output.append(item.to(self._device))
            elif isinstance(item,  dict):
                ks = item.keys()
                vs = self.toDevice(*item.values())
                item = dict(zip(ks, vs))
                output.append(item)
            elif isinstance(item, list):
                output.append(self.toDevice(*item))
            else:
                output.append(item)
        return output
    
    def get_log_metrics(self):
        output = dict()
        obj_, edge_ = self.eva_tool.get_mean_metrics()
        
        for k,v in obj_.items():
            output[k+'_node_cls'] = v
        for k,v in edge_.items():
            output[k+'_edge_cls'] = v
        return output
        