from ultralytics import YOLO
import numpy as np

def main():
    K = np.array([[600.0, 0.0, 320.0],
              [0.0, 600.0, 240.0],
              [0.0, 0.0, 1.0]])
    model = YOLO("/home/yishiuan/桌面/omni3d/runs/detect3d/train-2/weights/best.pt")
    results = model.predict("datasets/SUNRGBD/realsense/sh/2014_10_21-11_35_34-1311000041/image/0000067.jpg", K=K, show=False, save=True, conf=0.7)


if __name__ == "__main__":
    main()