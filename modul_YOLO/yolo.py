from ultralytics import YOLO
import os

import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

save_dir = os.path.join(os.getcwd(), 'model_yolo_gotowy/', 'v1')

def train_start():
    print('************************************************************ START ************************************************************')
    model = YOLO('yolo26s.pt')
    model.train(
        data='signs_1class.yaml',
        epochs=60,
        imgsz=1280,
        batch=8,
        device=0,
        amp=True,
        workers=8,
        cache=False,
        compile=False,
        mosaic=0.7,
        save_period=5,
        patience=15,
        plots=True,
        project=".",
        name=save_dir,
        exist_ok=True
    )

def train_resume():
    print('************************************************************ RESUME ************************************************************')
    last_weights = os.path.join(save_dir, "weights", "last.pt")
    model = YOLO(last_weights)
    model.train(resume=True)

train_resume()
