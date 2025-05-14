import json
import shutil
from pathlib import Path


def split_images_by_annotations(images_dir, train_json, val_json, output_dir):
    """
    Разделяет изображения на train/val папки на основе аннотаций COCO

    Аргументы:
        images_dir (str): Путь к папке с исходными изображениями
        train_json (str): Путь к файлу train.json
        val_json (str): Путь к файлу val.json
        output_dir (str): Корневая папка для выходной структуры
    """
    # Создаем выходные директории
    output_images = Path(output_dir) / 'images'
    (output_images / 'train').mkdir(parents=True, exist_ok=True)
    (output_images / 'val').mkdir(parents=True, exist_ok=True)

    # Обрабатываем train.json
    with open(train_json) as f:
        train_data = json.load(f)
    for img_info in train_data['images']:
        src = Path(images_dir) / img_info['file_name']
        dst = output_images / 'train' / img_info['file_name']
        shutil.copy(src, dst)

    # Обрабатываем val.json
    with open(val_json) as f:
        val_data = json.load(f)
    for img_info in val_data['images']:
        src = Path(images_dir) / img_info['file_name']
        dst = output_images / 'val' / img_info['file_name']
        shutil.copy(src, dst)


if __name__ == '__main__':
    # Конфигурация путей
    images_dir = 'datasets/HRSID/imagesjpg'  # Папка со всеми исходными изображениями
    train_json = 'datasets/HRSID/annotations/train2017.json'
    val_json = 'datasets/HRSID/annotations/test2017.json'
    output_dir = 'datasets/HRSID'  # Выходная корневая папка

    split_images_by_annotations(images_dir, train_json, val_json, output_dir)
    print("Изображения успешно разделены!")