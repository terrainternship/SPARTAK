## Описание
`Video Feature Analyzer` - это инструмент для анализа видеофайлов, который сравнивает фреймы видео по различным метрикам и выделяет ключевые кадры.

## Установка
Для начала работы с `Video Feature Analyzer` необходимо инициализировать его с указанием путей к видео и признакам:

```python
analyzer = VideoFeatureAnalyzer(video_path, features_path, debug=False, step=1, mode='inception')
```

### Параметры
- `video_path` Путь к папкам с видео.
- `features_path` Путь к посчитанным файлам признаков.
- `step` Размер шага, по которому идет сравнение (по умолчанию: 1).
- `mode` Модель расчета признаков. Доступные варианты: 'inception', 'clip'.

### Метод `process_and_analyze_frames`
```python
analyzer.process_and_analyze_frames(
	main_frame=5300,
	frames_count=300,
	extract_frames=True,
	start_folder='182 27.04.2023',
	break_folder='142 27.07.2022',
	compare_mode='cos'
)
```

#### Параметры
- `main_frame` Начальный фрейм первого видео для сравнения.
- `frames_count` Количество промежуточных кадров для следующего видео.
- `extract_frames`: Сохранять ли промежуточные кадры в отдельные папки.
- `start_folder` Папка, с которой начинается анализ.
- `break_folder`: Папка, на которой анализ останавливается.
- `compare_mode` Метод сравнения кадров (поддерживаются 'cos', 'euc', 'man').
- `frames_folder` Папка для промежуточных кадров
- `result_folder` папка для финальных кадров

## Примечания
- Использование GPU не требуется.
- Необходимость в большом объеме памяти отсутствует.
- Все данные предварительно посчитаны.
- Время анализа набора из 40 кадров для K13 составляет около 15 минут, если не сохранять все кадры по папкам.