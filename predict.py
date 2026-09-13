from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'D:\yolov12\runs\detect\train2\weights\best.pt')
    model.predict(source=r'D:\yolov12\datasets\Apple_Leaf_Dataset\test\images',
                  save=True,
                  show=False,
                  )