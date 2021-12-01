import argparse

import cv2
import os
import numpy as np
from numpy.core.numeric import outer
import torch
import pandas as pd

from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine
from tensorflow.keras.models import load_model

def parse_args():
    parser = argparse.ArgumentParser(description='Infer a detector')
    parser.add_argument('config', help='config file path')
    parser.add_argument('dataframe', help='dataframe contain labels')

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


def plot_result(result, imgfp, mask_model):
    bbox_color_map = {0: (0, 255, 0), 1: (0, 0, 255)}
    thickness = 1

    img = imread(imgfp)

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    h, w,_ = img.shape

    mask_cls = []
    for bbox in bboxes:
        bbox_int = bbox[:4].astype(np.int32)
        xmin, ymin, xmax, ymax = bbox_int
        
        # Classify mask 
        crop = img[ymin: ymax, xmin: xmax]
        crop = cv2.resize(crop,(128,128))
        crop = np.reshape(crop,[1,128,128,3])/255.0
        mask_result = mask_model.predict(crop).argmax()

        mask_cls.append(mask_result)

        left_top = (xmin, ymin)
        right_bottom = (xmax, ymax)
        cv2.rectangle(img, left_top, right_bottom, bbox_color_map[mask_result], thickness)
    
    cv2.imwrite(os.path.join('/content/new_train', os.path.basename(imgfp)), img)

    masked = None
    s =  list(map(lambda x: x == 0., mask_cls))
    if len(s) == 0:
        masked = None
    elif all(s):
        masked = 1
    elif any(s):
        masked = 0
    return masked


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    class_names = cfg.class_names

    mask_model = load_model(cfg.mask_checkpoint)
    engine, data_pipeline, device = prepare(cfg)

    dataframe = pd.read_csv(args.dataframe, index_col=0)
    dataframe['fname'] = dataframe['fname'].apply(lambda x: os.path.join('/content/images', x))

    labels = []
    for img_path in dataframe['fname'].tolist():
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
        label = plot_result(result, img_path, mask_model)
        labels.append(label)
    dataframe['real mask'] = labels
    dataframe.to_csv('new_train_meta.csv')

if __name__ == '__main__':
    main()