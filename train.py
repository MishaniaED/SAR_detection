import torch.cuda

from ultralytics import YOLO

# TODO: Подгрузить необученную модель YOLOv10 и запустить её на Каждом из датасетов с определенными настройками как в статьях
def main():
    # Загрузите модель YOLOv10
    model = YOLO('yolov10n.pt')  # или 'yolov10s.pt', 'yolov10m.pt' и т.д.
    device = 0 if torch.cuda.is_available() else 'cpu'

    # Обучите модель
    model.train(
        data='datasets/SAR_AirCraft-1.0/sar_aircraft-1.0.yaml',
        epochs=100,
        imgsz=640,
        batch=1,
        device=device  # Укажите GPU, если доступен
    )


if __name__ == '__main__':  # <-- Добавьте эту проверку
    main()