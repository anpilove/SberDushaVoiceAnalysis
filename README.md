<a id='ENG'></a>
<a href="#RUS"><img src='https://img.shields.io/badge/ENG -Go to RUS description-blue'></a>

# Sber Dusha Voice Analysis

---

![alt text](https://github.com/anpilove/SberDushaVoiceAnalysis/blob/main/streamlit/screenshot.png)

In this project, I developed models for mood recognition based on the Dusha by Sber dataset. Using various neural network architectures.

Additionally, a web application was developed using Streamlit, allowing users to test the models in various ways by uploading audio files or recording sound from the microphone. Users can also obtain statistics on audio messages, including information on mood, duration, and frequency of different moods.

## Project Organization

    ├── README.md                           <- The top-level README for developers using this project.
    ├── data
    │   └── samples                         <- angry, neutral, positive, sad audio.
    │
    ├── models                              <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                           <- Jupyter notebooks.
    │
    ├── streamlit                           <- Streamlit app.
    │
    ├── audio_classification.ipynb          <- Baseline notebooks.
    │
    └──requirements.txt                     <- The requirements file for reproducing the analysis environment, e.g.
                                            generated with `pip freeze > requirements.txt`

---

<a id='RUS'></a>
<a href="#ENG"><img src='https://img.shields.io/badge/RUS -Go to ENG description-blue'></a>

# Sber Dusha Voice Analysis

---

В этом проекте я разработал модели для распознавания настроения на основе датасета Dusha by Sber. Используя различные архитектуры нейронных сетей.

Кроме того, было разработано веб-приложение на платформе Streamlit, которое позволяет пользователям тестировать модели различными способами, загружая аудиофайлы или записывая звук с микрофона. Пользователи могут также получать статистику по аудиосообщениям, включая информацию о настроении, длительности и частоте использования различных настроений.
Мы можем так-же получить статистику по аудио сообщению.

## Организация проекта

    ├── README.md                           <- Основной README для разработчиков, использующих этот проект.
    ├── data
    │   └── samples                         <- злые, нейтральные, позитивные, грустные записи.
    │
    ├── models                              <- Обученные модели.
    │
    ├── notebooks                           <- Jupyter ноутбуки.
    │
    ├── streamlit                           <- Streamlit приложение.
    │
    ├── audio_classification.ipynb          <- Baseline ноутбук.
    │
    └──requirements.txt                     <-  Файл с требованиями для воспроизведения окружения анализа, например
                                                созданный с помощью `pip freeze > requirements.txt`
