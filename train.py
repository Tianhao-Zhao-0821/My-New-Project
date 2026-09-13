import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'D:\yolov12\ultralytics\cfg\models\v12\yolov12_dssa.yaml')
    model.train(data=r'D:\yolov12\datasets\Apple_Leaf_Dataset\data.yaml',
                imgsz=640,
                epochs=1,
                batch=16,
                workers=8,
                device='',
                optimizer='auto',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='dssa_test',
                single_cls=False,
                cache=False,
                )