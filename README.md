# small-data-cv-comparison

Объединил две лабораторные работы (классификация + детекция) в один репозиторий с архитектурой полноценного Python-пакета:
  - Переиспользуемый модуль src/cvlab/ вместо copy-paste кода по ноутбукам
  - CLI-скрипты для обучения и инференса
  - 70 автоматических тестов на те места, где баги молчат (bbox-конверсия, COCO-аудит, конфиг-генерация)
  - Двуязычная документация (README.md + README.ru.md, docs/results.md, docs/methodology.md)
  - CI, ruff, .gitignore, pyproject.toml
  - Исправил три дефекта:
    - D-FINE test-конфиг был побайтово идентичен train → --test-only оценивал валидацию
    - Grad-CAM никогда не запускался, хотя в документах утверждалось "подтвердил точность"
    - Утверждение о доминировании пропусков над ошибками классификации не имело подсчётов
  - Записал реальные метрики в results/*/metrics.csv
  - Выбрал 26 репрезентативных фигур из 89

## 🗂️ Dataset
Kaggle: https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/data

## 🙏 Special thanks
* [@oodwyn](https://github.com/oodwyn) & @emvoron — разметка данных в [![Label Studio](https://custom-icon-badges.demolab.com/badge/label%20studio-white?style=for-the-badge&logo=labelstudio)](#)