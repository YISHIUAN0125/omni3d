from ultralytics import YOLO


def main():
    model = YOLO("configs/yolo26n-cube.yaml", task="detect3d").load("pretrained/yolo26n.pt")
    model.train(
        data='configs/omni3d_38_classes.yaml',
        epochs=150,
        batch=16,
        imgsz=640,
        workers=4,
        device='0',
        close_mosaic=0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        amp=False,
        plots=True,
        save=True,
        save_period=-1,
        val=True,
        val_period=10,
        multi_scale=0.0,)


if __name__ == "__main__":
    main()