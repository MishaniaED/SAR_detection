import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os


def analyze_experiment(csv_path, output_dir):
    try:
        # Создаем папку для графиков
        # plots_dir = Path(output_dir) / 'training_not_augmented'
        plots_dir = Path(output_dir) / 'training_augmented'
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Читаем данные
        df = pd.read_csv(csv_path, sep=',', skipinitialspace=True)

        # Проверяем обязательные колонки
        required_columns = [
            'epoch', 'metrics/mAP50-95(B)',
            'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
            'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
            'metrics/precision(B)', 'metrics/recall(B)',
            'metrics/mAP50(B)', 'lr/pg0', 'lr/pg1', 'lr/pg2'
        ]

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Отсутствуют колонки: {missing}")

        # Строим графики
        experiment_name = csv_path.parent.name
        plot_metrics(df, experiment_name, plots_dir)

        # Находим лучшую эпоху
        best_row = df.loc[df['metrics/mAP50-95(B)'].idxmax()].copy()
        best_row['experiment_name'] = experiment_name

        return best_row

    except Exception as e:
        print(f"Ошибка обработки {csv_path}: {str(e)}")
        return None


def plot_metrics(df, exp_name, output_dir):
    plt.figure(figsize=(20, 15))

    # Графики потерь
    plt.subplot(3, 1, 1)
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
    plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
    plt.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
    plt.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
    plt.plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss')
    plt.plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss')
    plt.title(f'{exp_name} - Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Метрики качества
    plt.subplot(3, 1, 2)
    plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
    plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
    plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
    plt.title(f'{exp_name} - Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Learning Rate
    plt.subplot(3, 1, 3)
    plt.plot(df['epoch'], df['lr/pg0'], label='LR/pg0')
    plt.plot(df['epoch'], df['lr/pg1'], label='LR/pg1')
    plt.plot(df['epoch'], df['lr/pg2'], label='LR/pg2')
    plt.title(f'{exp_name} - Learning Rates')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    # Сохраняем графики
    plt.tight_layout()
    plt.savefig(output_dir / f'{exp_name}_metrics.png', dpi=150)
    plt.close()


def process_all_experiments(root_dir, output_dir):
    results = []
    csv_files = list(Path(root_dir).glob('**/results.csv'))

    if not csv_files:
        print("Не найдено ни одного файла results.csv!")
        return

    for csv_file in csv_files:
        print(f"Обработка: {csv_file.parent.name}")
        result = analyze_experiment(csv_file, output_dir)
        if result is not None:
            results.append(result)

    if results:
        # Создаем DataFrame
        result_df = pd.DataFrame(results)

        # Порядок колонок
        columns = [
            'experiment_name',
            'epoch',
            'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
            'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
            'metrics/precision(B)', 'metrics/recall(B)',
            'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
            'lr/pg0', 'lr/pg1', 'lr/pg2'
        ]

        # Сохраняем результаты
        output_csv = Path(output_dir) / 'experiments_augmented_summary.csv'
        # output_csv = Path(output_dir) / 'experiments_not_augmented_summary.csv'
        result_df[columns].to_csv(
            output_csv,
            index=False,
            sep=';',  # явно указываем разделитель
            encoding='utf-8-sig',  # для корректного отображения в Excel
            float_format='%.5f'  # формат чисел
        )
        print(f"Результаты сохранены: {output_csv}")
    else:
        print("Нет данных для сохранения")


# Использование:
process_all_experiments(
    # root_dir='runs/training/HRSID/augmented',
    # output_dir='runs_analize/training/HRSID'
    # root_dir='runs/training/SAR_AirCraft/augmented',
    # output_dir='runs_analize/training/SAR-AIRcraft'
    root_dir='runs/training/SSDD/augmented',
    output_dir='runs_analize/training/SSDD'

    # root_dir='runs/training/HRSID/not_augmented',
    # output_dir='runs_analize/training/HRSID'
    # root_dir='runs/training/SAR_AirCraft/not_augmented',
    # output_dir='runs_analize/training/SAR-AIRcraft'
    # root_dir='runs/training/SSDD/not_augmented',
    # output_dir='runs_analize/training/SSDD'
)