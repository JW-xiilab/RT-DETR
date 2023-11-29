"""by lyuwenyu
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from loguru import logger
import argparse
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

from src.core import YAMLConfig
from src.data import get_coco_api_from_dataset
from src.misc import dist
from src.solver import TASKS
from utils import PostProc, get_model_info

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', default='/root/dataset/coco2017/images/val2017', type=str,)
    parser.add_argument('--config', '-c', default='configs/rtdetr/rtdetr_r101vd_6x_coco.yml', type=str, )
    parser.add_argument('--resume', '-r', default='models/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth', type=str, )
    parser.add_argument('--out_dir', default='RTDETR_outputs')
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.1,
        help="the spread or amplitude of the Gaussian noise distribution",
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--save_txt",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument('--classes', nargs='*', default=[74], type=int)
    parser.add_argument('--iou_thres', type=float, default=0.5)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--amp', action='store_true', default=False,)
    return parser


class Model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        logger.info("Loading model")
        self.model = dist.warp_model(cfg.model.to(self.device), cfg.find_unused_parameters, cfg.sync_bn)
        self.criterion = cfg.criterion.to(self.device)
        self.postprocessor = PostProc(classes=cfg.classes, iou_thres=cfg.iou_thres)#cfg.postprocessor
        self.scaler = cfg.scaler
        self.ema = cfg.ema.to(self.device) if cfg.ema is not None else None 
        self.noise = cfg.noise_std
        if cfg.resume:
            print(f'resume from {cfg.resume}')
            self.resume(cfg.resume)
        self.model.eval()         
        logger.info("Model loaded")
        self.test_size = (640, 640)
        logger.info("Model Summary: {}".format(get_model_info(self.model, self.test_size)))
        self.transform = T.Compose([T.Resize(self.test_size),#self.test_size),
                                    T.ToImageTensor(),
                                    T.ConvertDtype()])
        
        #     self.model.cuda()
    # def forward(self, images, orig_target_sizes):
    #     outputs = self.model(images)
    #     return self.postprocessor(outputs, orig_target_sizes)

    def resume(self, path):
        '''load resume
        '''
        # for cuda:0 memory
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state)

    def load_state_dict(self, state):
        '''load state dict
        '''
        # TODO
        if getattr(self, 'last_epoch', None) and 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Loading last_epoch')

        if getattr(self, 'model', None) and 'model' in state:
            if dist.is_parallel(self.model):
                self.model.module.load_state_dict(state['model'])
            else:
                self.model.load_state_dict(state['model'])
            print('Loading model.state_dict')

        if getattr(self, 'ema', None) and 'ema' in state:
            self.ema.load_state_dict(state['ema'])
            print('Loading ema.state_dict')

        if getattr(self, 'optimizer', None) and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
            print('Loading optimizer.state_dict')

        if getattr(self, 'lr_scheduler', None) and 'lr_scheduler' in state:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            print('Loading lr_scheduler.state_dict')

        if getattr(self, 'scaler', None) and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
            print('Loading scaler.state_dict')

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            # img = cv2.imread(img)
            img = Image.open(img).convert('RGB')
        else:
            img_info["file_name"] = None
            return None, None
 
        # height, width = img.shape[:2]
        width, height = img.size
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        # ratio = min(self.test_size[0] / height, self.test_size[1] / width)
        # img_info["ratio"] = ratio
        img = self.transform(img)
        img = img.unsqueeze(0)
        orig_target_size = torch.tensor([width, height])
        if self.noise:
            noise = torch.randn_like(img) * self.noise
            img = img + noise
            img = torch.clamp(img, 0, 1)
        if self.device.type == 'cuda':
            img = img.cuda()
            orig_target_size = orig_target_size.cuda()

        with torch.no_grad():
            outputs = self.model(img)
        results = self.postprocessor(outputs, orig_target_size)
        
        return results, img_info

def main(args, ) -> None:
    '''main
    '''

    cfg = YAMLConfig(
        args.config,
        resume=args.resume
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('only support resume to load model.state_dict by now.')

    cfg.model.load_state_dict(state)
    if args.classes:
        cfg.classes = args.classes
    if args.iou_thres:
        cfg.iou_thres = args.iou_thres
    if args.noise_std:
        cfg.noise_std = args.noise_std

    model = Model(cfg)
    current_time = time.localtime()
    # args.out_dir = os.path.join(args.out_dir, str(args.classes[0]))
    image_demo(model, args.out_dir, args.path, current_time, args.classes, save_result=args.save_result, save_txt=args.save_txt)

def image_demo(model, out_dir, path, current_time, classes, save_result=None, save_txt=None):
    out_dir = Path(__file__).parents[-5].absolute() / out_dir 
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img_dir = Path(path)
    if img_dir.is_dir():
        files = get_image_list(path)
    elif img_dir.is_file(path):
        files = [path]
    else:
        print("The given directory/path doesn't seem to exist")
        raise
    files.sort()
    cnt = 0
    if save_result or save_txt:
        txt_folder = os.path.join(out_dir, 'labels', str(classes[0]))
        save_folder = os.path.join(
            txt_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        # vis_folder = os.path.join(save_folder, 'images')
        # save_folder = os.path.join(
        #     vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        # )
        
        os.makedirs(save_folder,  exist_ok=True)
    for image_name in tqdm(files, desc='Inferencing', total=len(files), leave=True):
        p = Path(image_name)
        outputs, _ = model.inference(image_name)
        if not isinstance(outputs, torch.Tensor):
            continue
        cnt += 1
        if save_txt:
            txt_file = os.path.join(save_folder, p.stem + '.txt')
            save_txt_file(txt_file, outputs) 
    logger.info(f'Results saved to {Path(txt_folder).resolve()}')
    logger.info(f'{cnt} labels saved to {Path(save_folder).resolve()}')

def save_txt_file(txt_file, result):
    texts = []
    for d in result:
        c, conf = int(d[0]), float(d[-1])
        line = (c, *d[1:-1].view(-1), conf)
        texts.append(('%g ' * len(line)).rstrip() % line)
    if texts:
        Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
        with open(txt_file, 'a') as f:
            f.writelines(text + '\n' for text in texts)

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


if __name__ == '__main__':
    args = make_parser().parse_args()

    main(args)
