from copy import deepcopy
from typing import Sequence

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision


class PostProc(nn.Module):
    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False, classes=None, iou_thres=0.25) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category 
        self.classes = classes
        self.iou_thres = iou_thres

    def forward(self, outputs, orig_target_sizes):

        logits, _boxes = outputs['pred_logits'], outputs['pred_boxes']       

        bbox_pred = torchvision.ops.box_convert(_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        output = [None for _ in range(len(logits))]
        if len(output) >= 2:
            print(1)

        if self.use_focal_loss:
            _scores = F.sigmoid(logits)
            _scores, index = torch.topk(_scores.flatten(1), self.num_top_queries, axis=-1)
            _labels = index % self.num_classes
            index = index // self.num_classes
            _boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        else:
            _scores = F.softmax(logits)[:, :, :-1]
            _scores, _labels = _scores.max(dim=-1)
            if _scores.shape[1] > self.num_top_queries:
                _scores, index = torch.topk(_scores, self.num_top_queries, dim=-1)
                _labels = torch.gather(_labels, dim=1, index=index)
                _boxes = torch.gather(_boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, _boxes.shape[-1]))

        mask = _scores > self.iou_thres

        detections = torch.cat((_labels.unsqueeze(-1), _boxes, _scores.unsqueeze(-1)), 2)
        detections = detections[mask]
        
        if self.remap_mscoco_category:
            from ..src.data.coco import mscoco_label2category
            detections[:,-1] = torch.tensor([mscoco_label2category[int(x.item())] for x in detections[:,0].flatten()])\
                .to(_boxes.device).reshape(detections[:,0].shape)

        if self.classes is not None:
            detections = detections[(detections[:, 0] == torch.tensor(self.classes, device=detections.device))]
        
        if not detections.size(0):
            return None

        results = []
        for _, dets in enumerate(detections):
            
            # dets[1:-1] = self.convert_bbox_to_standard(dets, orig_target_sizes)

            # dets[1:-1] = x1, y1, x2, y2
            # # TODO
            # # if i == 1:
            # #     print(1)
            # for _det in dets:
            #     try:
            #         if output[0] is None:
            #             output[0] = _det.unsqueeze(0)
            #     except IndexError:
            #         print(1)
                # else:
            results.append(dets)# = torch.cat((output[i], _det.unsqueeze(0)))
                    # result = dict(labels=lab, boxes=box, scores=sco)
                    # results.append(result)
        
        if not results:
            return None
        return torch.stack(results)

    def convert_bbox_to_standard(bbox, orig_target_size):
        x1, y1, x2, y2 = bbox[1:-1]
        width, height = orig_target_size



def get_model_info(model: nn.Module, tsize: Sequence[int]) -> str:
    from thop import profile

    stride = 640
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info