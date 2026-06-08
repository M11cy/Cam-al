# Быстрое обучение модели под эту камеру

Схема классов для первого эксперимента:

```text
person
phone
other_object
```

`other_object` - это предметы, которые похожи на телефон, но не должны запускать тревогу: вейп, расчёска, банка, упаковка, ключи, маленькая коробка.

## 1. Собрать кадры

Автоматически собрать 100 кадров с камеры:

```bash
.venv/bin/python tools/capture_frames.py --camera 0 --count 100 --interval 1
```

Если камера другая:

```bash
.venv/bin/python tools/capture_frames.py --camera 1 --count 100 --interval 1
```

Можно собирать вручную: нажимать `Space`, чтобы сохранить кадр.

```bash
.venv/bin/python tools/capture_frames.py
```

## 2. Разметить кадры

```bash
.venv/bin/python tools/annotate_dataset.py
```

Клавиши:

- `p` - выделить сотрудника;
- `t` - выделить телефон;
- `o` - выделить похожий предмет как `other_object`;
- `u` - отменить последнюю рамку;
- `n` - сохранить кадр и перейти дальше;
- `s` - сохранить кадр без объектов;
- `q` - выйти.

Для первого теста хватит 50-150 размеченных кадров. Важно добавить кадры, где есть предметы, похожие на телефон.

## 3. Обучить модель

```bash
.venv/bin/python tools/train_custom_model.py --epochs 50 --imgsz 960 --batch 4
```

Готовая модель появится примерно здесь:

```text
runs/detect/shop-phone/weights/best.pt
```

Если запускать обучение несколько раз, папка может получить номер:

```text
runs/detect/shop-phone2/weights/best.pt
```

## 4. Подключить модель

Скопируйте `best.pt` в папку `models`:

```bash
mkdir -p models
cp runs/detect/shop-phone/weights/best.pt models/shop-camera-phone.pt
```

В `app.py` замените:

```python
MODEL_NAME = "yolov8s.pt"
```

на:

```python
MODEL_NAME = "models/shop-camera-phone.pt"
```

После этого приложение будет использовать локальную модель, обученную под эту камеру.
