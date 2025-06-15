import itertools
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

import pandas as pd
from matplotlib import patches
from tqdm import tqdm


# TODO: guided filter

def apply_lee_filter(image, window_size=3, MV=0.25):
    """
    Фильтр Ли для подавления мультипликативного спекл-шума

    Параметры:
    image - входное изображение
    window_size - размер окна фильтра (нечетное число)
    MV - дисперсия мультипликативного шума (экспериментально подбирается)

    Формула:
    filtered = LM + K * (PC - LM)
    K = LV / (LM² * MV + LV)
    """
    # Конвертация и нормализация
    image = np.float32(image) / 255.0

    # Вычисление локального среднего
    mean = cv2.blur(image, (window_size, window_size))

    # Вычисление локальной дисперсии
    mean_sq = cv2.blur(image ** 2, (window_size, window_size))
    variance = np.maximum(mean_sq - mean ** 2, 1e-6)

    # Расчет коэффициента усиления
    # k = variance / (variance + MV)
    k = variance / ((mean**2 * MV) + variance)

    # Применение фильтра
    filtered = mean + k * (image - mean)

    # Денормализация и конвертация
    return np.uint8(np.clip(filtered * 255, 0, 255))


def process_folder(input_folder, output_folder, window_size=3, MV=0.5):
    """Обработка всех изображений в папке"""
    # Создаем выходную папку если не существует
    os.makedirs(output_folder, exist_ok=True)

    # Получаем список файлов с поддержкой вложенных директорий
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    file_list = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(supported_formats):
                file_list.append(os.path.join(root, file))

    # Обрабатываем файлы с прогресс-баром
    for input_path in tqdm(file_list, desc="Processing images"):
        try:
            # Создаем относительный путь для сохранения
            rel_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Читаем и обрабатываем изображение
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)  # SAR обычно grayscale
            if img is None:
                raise ValueError(f"Не удалось прочитать файл {input_path}")

            # Применяем фильтр Ли
            filtered_img = apply_lee_filter(img, window_size, MV)

            # Сохраняем результат
            cv2.imwrite(output_path, filtered_img)

        except Exception as e:
            print(f"\nОшибка при обработке {input_path}: {str(e)}")


def parameters_test(input_path, output_folder, window_sizes=[3, 5, 7, 9], MV=np.arange(0.1, 0.55, 0.05)):
    """Тестирование разных параметров фильтра для одного изображения"""
    # Читаем изображение
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось прочитать файл {input_path}")

    # Создаем выходную папку
    os.makedirs(output_folder, exist_ok=True)

    # Генерируем все комбинации параметров
    params = list(itertools.product(window_sizes, MV))
    results = []

    # Обрабатываем все комбинации
    for ws, mv in tqdm(params, desc="Тестирование параметров"):
        try:
            # Применяем фильтр
            filtered = apply_lee_filter(img, window_size=ws, MV=mv)
            # Контрастное усиление
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # filtered = clahe.apply(filtered)
            # result = cv2.bilateralFilter(filtered, 5, 75, 75)

            # Формируем имя файла
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_name = f"{base_name}_ws{ws}_nv{mv:.2f}.png"
            output_path = os.path.join(output_folder, output_name)

            # Сохраняем результат
            cv2.imwrite(output_path, filtered)

            # Выделение ROI
            filtered_roi_water = filtered[600:700, 400:500]  # Водная поверхность
            filtered_roi_ship = filtered[515:600, 575:620]  # Корпус корабля
            orig_roi_water = img[600:700, 400:500]
            orig_roi_ship = img[515:600, 575:620]

            # plot_roi(filtered, 'ROI_filtered_water', 600, 700, 400, 500)
            # plot_roi(img, 'ROI_orig_ship', 515, 600, 575, 620)
            # plot_roi(filtered, 'ROI_filtered_water', 600, 700, 400, 500)
            # plot_roi(img, 'ROI_orig_ship', 515, 600, 575, 620)

            # Рассчитываем метрики
            metrics = {
                'filename': output_name,  # Добавляем имя файла первым полем
                'window_size': ws,
                'MV': f"{mv:.2f}",
                'ENL_original_water': calculate_enl(orig_roi_water),
                'ENL_filtered_water': calculate_enl(filtered_roi_water),
                'SSI_water': calculate_ssi(orig_roi_water, filtered_roi_water),
                'EPI_ship': calculate_epi(orig_roi_ship, filtered_roi_ship),
                'STD_original_ship': np.std(orig_roi_ship),
                'STD_filtered_ship': np.std(filtered_roi_ship),
                'Mean_original_water': np.mean(orig_roi_water),
                'Mean_filtered_water': np.mean(filtered_roi_water)
            }
            results.append(metrics)

        except Exception as e:
            print(f"\nОшибка для параметров ws={ws}, nv={mv}: {str(e)}")

    # Сохраняем метрики в CSV
    df = pd.DataFrame(results)

    # Упорядочиваем колонки
    columns_order = [
        'filename',
        'window_size',
        'MV',
        'ENL_original_water',
        'ENL_filtered_water',
        'SSI_water',
        'EPI_ship',
        'STD_original_ship',
        'STD_filtered_ship',
        'Mean_original_water',
        'Mean_filtered_water'
    ]

    df[columns_order].to_csv(os.path.join(output_folder, 'metrics.csv'), index=False, sep=';')


def calculate_enl(image_roi):
    """
    image_roi: однородная область изображения (например, водная поверхность)
    """
    mean = np.mean(image_roi)
    std = np.std(image_roi)
    return (mean ** 2) / (std ** 2)


def calculate_ssi(original_roi, filtered_roi):
    return np.std(filtered_roi) / np.std(original_roi)


def calculate_epi(original, filtered):
    sobel_orig = cv2.Sobel(original, cv2.CV_64F, 1, 1)
    sobel_filt = cv2.Sobel(filtered, cv2.CV_64F, 1, 1)
    return np.corrcoef(sobel_orig.flatten(), sobel_filt.flatten())[0, 1]


def plot_roi(image, title, x1, x2, y1, y2):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image, cmap='gray')
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                             edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.title(title)
    plt.show()


# sample = cv2.imread('datasets/HRSID/processed_images/testing/P0001_0_800_8400_9200_ws3_nv0.05.png', 0)
#
# # Для оригинального изображения:
# plot_roi(sample, "Original Image ROIs",
#         600,700, 400,500)
# plot_roi(sample, "Original Image ROIs",
#         515,600,575,620)

# Генерация тестового шума
# noise = np.random.normal(0.5, 0.3, (100,100)).astype(np.float32)
# noise_uint8 = np.uint8(np.clip(noise*255, 0, 255))
#
# # Применение фильтра
# filtered = apply_lee_filter(noise_uint8, 7, 0.1)
#
# # Анализ результатов
# print(f"STD до: {np.std(noise_uint8):.2f}")
# print(f"STD после: {np.std(filtered):.2f}")
# print(f"PSNR: {cv2.PSNR(noise_uint8, filtered):.2f} dB")
#
# # Визуализация
# cv2.imshow("Original Noise", noise_uint8)
# cv2.imshow("Filtered", filtered)
# cv2.waitKey(0)

if __name__ == "__main__":
    input_folder = "datasets/HRSID/images_true/train"
    output_folder = "datasets/HRSID/images/testing_Le+fixed"

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_formats)]

    if not files:
        print("Нет изображений для тестирования!")
        exit(1)

    test_image = os.path.join(input_folder, files[1])
    print(f"Тестирование на изображении: {test_image}")

    # Запускаем тест
    parameters_test(
        input_path=test_image,
        output_folder=output_folder,
        window_sizes=[3, 5, 7, 9],  # Можно изменить значения
        MV=np.arange(0.05, 0.55, 0.05)  # От 0.1 до 0.5 с шагом 0.05
    )

    # print(f"Начата обработка...")
    # process_folder(
    #     input_folder=input_folder,
    #     output_folder=output_folder,
    #     window_size=3,
    #     MV=0.05
    # )
    # print(f"\nОбработка завершена! Результаты сохранены в {output_folder}")
