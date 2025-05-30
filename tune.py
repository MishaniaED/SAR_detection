import torch

from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("ultralytics/cfg/models/12/yolo12.yaml")
device = 0 if torch.cuda.is_available() else 'cpu'
# Define search space
search_space = {
    "lr0": (1e-5, 1e-1),
    "lrf": (0.0, 1.0),
    "momentum": (0.6, 0.98),
    "weight_decay": (0.0, 0.001),
    "warmup_epochs": (0.0, 5.0),
    "warmup_momentum": (0.0, 0.95),
    "box": (0.02, 8.5),
    "cls": (0.2, 6.0)
}

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(
    data="datasets/Official-SSDD-OPEN/ssdd.yaml",
    project='runs/tuning/SSDD',  # Основная папка для экспериментов
    name='yolov12n_augment_processed_Le_SIoU',  # Имя подпапки для конкретного запуска
    exist_ok=True,  # Продолжить обучение в существующей папке
    space=search_space,
    epochs=50,
    iterations=15,
    device=device,
    pretrained=False,

    optimizer="SGD",
    batch=16,

    auto_augment=None,
    hsv_h=0.0,    # Позволяет регулировать оттенок изображения на малую долю цветового круга, внося разнообразие в цветовую гамму.
    hsv_s=0.0,  # Немного изменяет насыщенность изображения, влияя на интенсивность цветов.
    hsv_v=0.0,  # Изменяет значение (яркость) изображения на незначительную величину
    degrees=0.0,  # Произвольный поворот изображения в заданном диапазоне градусов
    translate=0.0,  # Преобразует изображение по горизонтали и вертикали на долю размера изображения
    scale=0.0,  # Масштабирует изображение
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,  # Переворачивает изображение слева направо с заданной вероятностью
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    close_mosaic=10,  # Отключает аугментацию mosaic за 10 эпох до окончания обучения
    cutmix=0.0,

    plots=False,
    save=False,
    val=False,
)


model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")
model.tune(
    data="datasets/Official-SSDD-OPEN/ssdd.yaml",
    project='runs/tuning/SSDD',  # Основная папка для экспериментов
    name='yolov11n_augment_processed_Le_SIoU',  # Имя подпапки для конкретного запуска
    exist_ok=True,  # Продолжить обучение в существующей папке
    space=search_space,
    epochs=50,
    iterations=15,
    device=device,
    pretrained=False,

    optimizer="SGD",
    batch=16,

    auto_augment=None,
    hsv_h=0.0,    # Позволяет регулировать оттенок изображения на малую долю цветового круга, внося разнообразие в цветовую гамму.
    hsv_s=0.0,  # Немного изменяет насыщенность изображения, влияя на интенсивность цветов.
    hsv_v=0.0,  # Изменяет значение (яркость) изображения на незначительную величину
    degrees=0.0,  # Произвольный поворот изображения в заданном диапазоне градусов
    translate=0.0,  # Преобразует изображение по горизонтали и вертикали на долю размера изображения
    scale=0.0,  # Масштабирует изображение
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,  # Переворачивает изображение слева направо с заданной вероятностью
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    close_mosaic=10,  # Отключает аугментацию mosaic за 10 эпох до окончания обучения
    cutmix=0.0,

    plots=False,
    save=False,
    val=False,
)


model = YOLO("ultralytics/cfg/models/v10/yolov10n.yaml")
model.tune(
    data="datasets/Official-SSDD-OPEN/ssdd.yaml",
    project='runs/tuning/SSDD',  # Основная папка для экспериментов
    name='yolov10n_augment_processed_Le_SIoU',  # Имя подпапки для конкретного запуска
    exist_ok=True,  # Продолжить обучение в существующей папке
    space=search_space,
    epochs=50,
    iterations=15,
    device=device,
    pretrained=False,

    optimizer="SGD",
    batch=16,

    auto_augment=None,
    hsv_h=0.0,    # Позволяет регулировать оттенок изображения на малую долю цветового круга, внося разнообразие в цветовую гамму.
    hsv_s=0.0,  # Немного изменяет насыщенность изображения, влияя на интенсивность цветов.
    hsv_v=0.0,  # Изменяет значение (яркость) изображения на незначительную величину
    degrees=0.0,  # Произвольный поворот изображения в заданном диапазоне градусов
    translate=0.0,  # Преобразует изображение по горизонтали и вертикали на долю размера изображения
    scale=0.0,  # Масштабирует изображение
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,  # Переворачивает изображение слева направо с заданной вероятностью
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    close_mosaic=10,  # Отключает аугментацию mosaic за 10 эпох до окончания обучения
    cutmix=0.0,

    plots=False,
    save=False,
    val=False,
)


model = YOLO("ultralytics/cfg/models/SFS-CNet/sfs_cnet.yaml")
model.tune(
    data="datasets/Official-SSDD-OPEN/ssdd.yaml",
    project='runs/tuning/SSDD',  # Основная папка для экспериментов
    name='sfs_cnet_not_augment_processed_Le',  # Имя подпапки для конкретного запуска
    exist_ok=True,  # Продолжить обучение в существующей папке
    space=search_space,
    epochs=50,
    iterations=15,
    device=device,
    pretrained=False,

    optimizer="SGD",
    batch=16,

    auto_augment=None,
    hsv_h=0.0,    # Позволяет регулировать оттенок изображения на малую долю цветового круга, внося разнообразие в цветовую гамму.
    hsv_s=0.0,  # Немного изменяет насыщенность изображения, влияя на интенсивность цветов.
    hsv_v=0.0,  # Изменяет значение (яркость) изображения на незначительную величину
    degrees=0.0,  # Произвольный поворот изображения в заданном диапазоне градусов
    translate=0.0,  # Преобразует изображение по горизонтали и вертикали на долю размера изображения
    scale=0.0,  # Масштабирует изображение
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,  # Переворачивает изображение слева направо с заданной вероятностью
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    close_mosaic=10,  # Отключает аугментацию mosaic за 10 эпох до окончания обучения
    cutmix=0.0,

    plots=False,
    save=False,
    val=False,
)

# Tune hyperparameters on COCO8 for 30 epochs
# model.tune(
#     data="datasets/Official-SSDD-OPEN/ssdd.yaml",
#     project='runs/tuning/SSDD',  # Основная папка для экспериментов
#     name='yolov12n_augment',  # Имя подпапки для конкретного запуска
#     exist_ok=True,  # Продолжить обучение в существующей папке
#     space=search_space,
#     epochs=30,
#     iterations=50,
#     device=device,
#     pretrained=False,
#
#     optimizer="SGD",
#     batch=16,
#
#     hsv_h=0.015, # Позволяет регулировать оттенок изображения на малую долю цветового круга, внося разнообразие в цветовую гамму.
#     hsv_s=0.5,  # Немного изменяет насыщенность изображения, влияя на интенсивность цветов.
#     hsv_v=0.3,  # Изменяет значение (яркость) изображения на незначительную величину
#     degrees=10,  # Произвольный поворот изображения в заданном диапазоне градусов
#     translate=0.1,  # Преобразует изображение по горизонтали и вертикали на долю размера изображения
#     # scale=0.5,  # Масштабирует изображение
#     shear=0.0,
#     perspective=0.0,
#     flipud=0.0,
#     fliplr=0.5,  # Переворачивает изображение слева направо с заданной вероятностью
#     mosaic=1.0,    # Объединяет четыре обучающих изображения в одно, имитируя различные композиции сцен и взаимодействие объектов.
#     mixup=0.0,
#     copy_paste=0.0,
#     close_mosaic=10,  # Отключает аугментацию mosaic за 10 эпох до окончания обучения
#     cutmix=0.0,
#
#     plots=False,
#     save=False,
#     val=False,
# )