# JPEG_PARS

Утилита для пакетной сортировки JPEG-чертежей и OCR-парсинга размеров через единый графический интерфейс.

Для template OCR теперь приоритетно используется `PaddleOCR`, а `Tesseract` оставлен как fallback.

## Что умеет

- группирует JPEG по визуальному, структурному и OCR-сходству;
- раскладывает результаты по папкам `group_001`, `group_002`, ...;
- показывает группы и превью файлов в GUI;
- позволяет создать шаблон по одному изображению;
- позволяет мышкой размечать прямоугольные области поиска;
- позволяет сохранять и загружать шаблоны областей в JSON;
- поддерживает zoom колесом мыши и pan средней кнопкой мыши в окне шаблона;
- парсит значения из всех JPEG выбранной папки;
- показывает confidence OCR в интерфейсе;
- экспортирует таблицу значений и confidence в Excel `.xlsx`.

## Установка

```powershell
cd C:\pdf_ingest\JPEG_PARS
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

## Tesseract OCR

Для OCR нужен установленный Tesseract.

Типичный путь на Windows:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Если `tesseract.exe` уже есть в `PATH`, отдельный путь указывать не нужно.

## PaddleOCR

Для лучшего распознавания зон с размерами, мелкими символами и вертикальным текстом template-парсер теперь использует `PaddleOCR` как основной backend.

Минимальная установка по официальной схеме:

```powershell
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install paddleocr
```

Если `PaddleOCR` не установлен, template OCR автоматически откатится на `Tesseract`.

В проекте также отключена проверка доступности model hosters через переменную
`PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`, чтобы GUI не подвисал на старте.
При первом реальном использовании `PaddleOCR` модели все равно могут скачаться один раз.

## Запуск

Графический интерфейс:

```powershell
jpeg-pars-gui
```

CLI группировки:

```powershell
jpeg-pars `
  --input "C:\data\drawings" `
  --output "C:\data\drawings_sorted" `
  --similarity 82 `
  --ocr-mode auto
```

С явным путем к Tesseract:

```powershell
jpeg-pars `
  --input "C:\data\drawings" `
  --output "C:\data\drawings_sorted" `
  --similarity 82 `
  --ocr-mode required `
  --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe" `
  --ocr-lang "eng+rus" `
  --ocr-psm 6
```

## Сценарии GUI

### Группировка

- выбрать папку с JPEG;
- выбрать папку результата;
- задать порог сходства `1..100`;
- нажать `Запустить группировку`;
- просматривать дерево групп и превью файлов.

### Шаблон OCR

- нажать `Создать шаблон` и выбрать JPEG;
- при необходимости нажать `Загрузить шаблон` и открыть ранее сохраненный JSON;
- нажать `Выбрать границы поиска`;
- мышкой нарисовать прямоугольную область поверх изображения;
- справа появится строка с цветом, атрибутом и значением;
- имя атрибута по умолчанию идет `A..X`, затем `A1`, `B1` и так далее;
- колесо мыши меняет масштаб, средняя кнопка мыши двигает изображение;
- выбрать папку с JPEG;
- нажать `Запустить парсинг`;
- нажать `Сохранить шаблон` или `Экспорт в Excel`.

## Аргументы CLI

- `--input` путь к папке с изображениями
- `--output` папка для результатов
- `--similarity` порог сходства от `1` до `100`
- `--mode` `copy` или `move`
- `--recursive` искать изображения рекурсивно
- `--min-group-size` минимальный размер группы
- `--ocr-mode` `auto`, `off` или `required`
- `--tesseract-cmd` полный путь к `tesseract.exe`
- `--ocr-lang` языки OCR, например `eng` или `eng+rus`
- `--ocr-psm` режим page segmentation для Tesseract

## Как считается сходство

Итоговый score строится из нескольких слоев:

- `pHash` и `dHash` для формы изображения;
- edge-map и проекции линий;
- структурные признаки листа:
  - плотность заполнения
  - bbox содержимого
  - сетка заполнения `4x4`
  - число значимых компонентов
- OCR-признаки:
  - пересечение токенов
  - вертикальное распределение текстовых блоков

## GitHub

```powershell
cd C:\pdf_ingest\JPEG_PARS
git add .
git commit -m "Add desktop GUI and template OCR workflow"
git push
```
