import argparse

import cv2
import os
import numpy as np
from numpy.core.numeric import outer
import torch
import pandas as pd
import sklearn
from torchvision.transforms import ToTensor

from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine


def parse_args():
    parser = argparse.ArgumentParser(description='Infer a detector')
    parser.add_argument('--config', help='Config file path')
    parser.add_argument('--dataframe', help='Dataframe contain labels')
    parser.add_argument('--dataroot', help='Root contain image')
    parser.add_argument('--saving_dir', help='Directory to save result')

    args = parser.parse_args()
    return args


def prepare(cfg):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    engine = build_engine(cfg.infer_engine)

    engine.model.to(device)
    load_weights(engine.model, cfg.weights.filepath)

    data_pipeline = Compose(cfg.data_pipeline)
    return engine, data_pipeline, device


def crop2tensor(crop, device):
    crop = cv2.resize(crop, (128, 128)) / 255.
    crop = crop.astype(np.float32) 
    crop = torch.unsqueeze(ToTensor()(crop), axis=0).to(device)

    return crop


def plot_result(result, imgfp, mask_model, device, saving_dir):
    bbox_color_map = {0: (0, 255, 0), 1: (0, 0, 255)}
    thickness = 2

    img = imread(imgfp)

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    h, w, _ = img.shape

    crop_tensors = []
    mask_cls = []
    for bbox in bboxes:
        bbox_int = bbox[:4].astype(np.int32)
        xmin, ymin, xmax, ymax = bbox_int

        # Classify mask
        crop = img[ymin: ymax, xmin: xmax]
        crop_tensors.append(crop2tensor(crop, device))

    crop_tensors = torch.cat(crop_tensors, dim=0)
    crop_batches = torch.split(crop_tensors, 8)
    for crop_batch in crop_batches:
        mask_results = mask_model(crop_batch)[0]

        for mask_result in mask_results: 
            mask_result = mask_result.detach().cpu().numpy()[0] > 0.5    
            mask_cls.append(mask_result)

            left_top = (xmin, ymin)
            right_bottom = (xmax, ymax)
            cv2.rectangle(img, left_top, right_bottom, bbox_color_map[mask_result], thickness)

    cv2.imwrite(os.path.join(saving_dir, os.path.basename(imgfp)), img)

    s = list(map(lambda x: x == 0., mask_cls))
    if any(s):
        return 0
    else:
        return 1


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    class_names = cfg.class_names

    mask_model = torch.load(cfg.mask_checkpoint)
    engine, data_pipeline, device = prepare(cfg)

    df = pd.read_csv(args.df, index_col=0)
    df_result = pd.DataFrame(columns=['fname', 'mask'])
    if 'mask' in df.columns:
        df_comp = pd.DataFrame(columns=['fname', 'mask', 'gt_mask'])

    for fname in df['fname'].tolist():
        img_path = os.path.join(args.dataroot, fname)
        data = dict(img_info=dict(filename=img_path), img_prefix=None)

        data = data_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if device != 'cpu':
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            # just get the actual data from DataContainer
            data['img_metas'] = data['img_metas'][0].data
            data['img'] = data['img'][0].data
        result = engine.infer(data['img'], data['img_metas'])[0]

        label = plot_result(result, img_path, mask_model, device, saving_dir)
        df_result.append({'fname': fname, 'mask': label}, ignore_index=True)
        if 'mask' in df.columns:
            gt_mask = df.loc[df['fname'] == fname]['mask'] 
            if not np.isnan(gt_mask):
                df_comp.append({'fname': fname, 'mask': label, 'gt_mask': gt_mask)}, ignore_index=True)
    
    new_filename = os.path.splitext(args.df)[0] + '_result'
    df_result.to_csv(args.saving_dir, new_filename)

    pred = df_comp['mask'].tolist()
    gt = df_comp['gt_mask'].tolist()
    print(sklearn.metrics.accuracy_score(pred, gt))


if __name__ == '__main__':
    main()