# Backpack Finder (practice)

Вариант 16: поиск забытых рюкзаков в аэропорту.

## Возможности
- Детекция объектов (YOLO, предобученные веса)
- (Опционально) Instance-сегментация (Mask R-CNN, torchvision weights)
- Классификация кропов (ResNet50/EfficientNet, ImageNet) + прикладной `bag_type`
- Веб-интерфейс (Streamlit): Детекция / Сегментация / История / Отчёты
- История запросов в SQLite
- Генерация отчётов: PDF + Excel

## Установка
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Запуск API
```bash
uvicorn app.main:app --reload
```
Проверка: http://127.0.0.1:8000/health

## Запуск UI (Streamlit)
```bash
streamlit run ui/streamlit_app.py
```

## Настройки
В `app/config.py` можно включать/выключать модули:
- `ENABLE_SEGMENTATION`
- `ENABLE_CLASSIFICATION`
- `CLASSIFIER_MODEL_NAME` = `"resnet50"` или `"efficientnet_b0"`
