import os
from pathlib import Path

# val_labels = "datasets/SAR_AirCraft-1.0/labels/val"
# empty_files = []
#
# for file in os.listdir(val_labels):
#     if file.endswith(".txt"):
#         path = os.path.join(val_labels, file)
#         if os.stat(path).st_size == 0:
#             empty_files.append(file)
#
# print(f"Найдено пустых аннотаций: {len(empty_files)}")
# print(empty_files)



# Пути к папкам
images_dir = Path("datasets/SAR_AirCraft-1.0/images/val")
labels_dir = Path("datasets/SAR_AirCraft-1.0/labels/val")

# Получаем имена файлов без расширений
image_files = {f.stem for f in images_dir.glob("*") if f.suffix in [".jpg", ".png", ".jpeg"]}
label_files = {f.stem for f in labels_dir.glob("*.txt")}

# Находим расхождения
missing_in_labels = image_files - label_files
missing_in_images = label_files - image_files

print(f"Файлов в images/val: {len(image_files)}")
print(f"Файлов в labels/val: {len(label_files)}")
print("\nОтсутствует в labels/val:", missing_in_labels)
print("Отсутствует в images/val:", missing_in_images)