from openai import OpenAI
import streamlit as st
from st_audiorec import st_audiorec
import torch
import torchaudio
import torchaudio.transforms as T
import os
import random
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, Wav2Vec2FeatureExtractor, \
    HubertForSequenceClassification

import matplotlib.pyplot as plt



@st.cache_resource
def load_model_whisper():
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float32, use_safetensors=True
    )

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


@st.cache_resource
def load_model_hubert():
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertForSequenceClassification.from_pretrained(
        "xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned")
    return model, feature_extractor


def process_audio(audio):
    st.subheader("Анализ аудио")
    recorded_audio, sample_rate = torchaudio.load(audio, normalize=True)
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    recorded_audio = resampler(recorded_audio)

    if recorded_audio.shape[0] > 1:
        recorded_audio = torch.mean(recorded_audio, dim=0, keepdim=True)

    with st.spinner("Транскрипция"):
        st.write("Транскрипция: ", transcript_audio(recorded_audio))

    with st.spinner("Анализ тональности"):
        st.write("Предсказанная эмоция: ", sentiment_audio(recorded_audio))

    st.write("Количество каналов:", recorded_audio.shape[0])
    st.write("Частота дискретизации:", 16000)
    st.write("Длина записи:", recorded_audio.size(1) / 16000, "секунд")

    plt.figure(figsize=(10, 4))
    plt.plot(torch.arange(0, recorded_audio.size(1)) / 16000, recorded_audio[0].numpy())
    plt.title('Временной график аудио')
    plt.xlabel('Время (секунды)')
    plt.ylabel('Амплитуда')
    st.pyplot(plt)

    plt.figure(figsize=(10, 4))
    spec_transform = T.Spectrogram()
    specgram = spec_transform(recorded_audio)
    plt.imshow(specgram[0].numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.title('Спектрограмма')
    plt.xlabel('Время (секунды)')
    plt.ylabel('Частота (Гц)')
    plt.colorbar(label='Уровень мощности (дБ)')
    st.pyplot(plt)

    segment_duration = 1

    emotions_per_second = []
    timestamps = []

    for i in range(0, recorded_audio.size(1), sample_rate * segment_duration):
        segment = recorded_audio[:, i:i + sample_rate * segment_duration]
        emotion = sentiment_audio(segment)
        emotions_per_second.append(emotion)
        timestamps.append(i / sample_rate)

    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, emotions_per_second, marker='o', linestyle='-')
    plt.title('Эмоции на каждую секунду')
    plt.xlabel('Время (секунды)')
    plt.ylabel('Эмоция')
    plt.grid(True)
    st.pyplot(plt)


def transcript_audio(audio):
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_whisper,
        tokenizer=processor_whisper.tokenizer,
        feature_extractor=processor_whisper.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch.float32,
    )
    result = pipe(audio.squeeze().numpy(), generate_kwargs={"language": "russian"})
    transcription_text = result["text"]
    return transcription_text


def sentiment_audio(audio):
    num2emotion = {0: 'neutral', 1: 'angry', 2: 'positive', 3: 'sad', 4: 'other'}
    inputs = processor_hubert(
        audio,
        sampling_rate=processor_hubert.sampling_rate,
        return_tensors="pt",
        padding=True,
        max_length=16000 * 10,
        truncation=True
    )
    logits = model_hubert(inputs['input_values'][0]).logits
    predictions = torch.argmax(logits, dim=-1)
    predicted_emotion = num2emotion[predictions.numpy()[0]]
    return predicted_emotion


st.set_page_config(
    page_title='Audio Sentiment',
    layout="wide",
    initial_sidebar_state="expanded",
)




st.subheader("Загрузка аудио")

model_whisper, processor_whisper = load_model_whisper()
model_hubert, processor_hubert = load_model_hubert()

selected_option = st.selectbox('Выберите способ', ["Demo file", "Upload file", "Record file", "Audio GPT"])

if selected_option == "Demo file":
    demo_options = ["Random angry audio", "Random sad audio", "Random neutral audio", "Random positive audio"]
    selected_demo_option = st.selectbox("Выберите опцию", demo_options, index=None)
    if selected_demo_option == "Random angry audio":
        folder_path = "./angry"
        file_list = [file for file in os.listdir(folder_path)]
        audio = random.choice(file_list)

    elif selected_demo_option == "Random sad audio":
        folder_path = "./sad"
        file_list = [file for file in os.listdir("./sad")]
        audio = random.choice(file_list)

    elif selected_demo_option == "Random neutral audio":
        folder_path = "./neutral"
        file_list = [file for file in os.listdir("./neutral")]
        audio = random.choice(file_list)

    elif selected_demo_option == "Random positive audio":
        folder_path = "./positive"
        file_list = [file for file in os.listdir("./positive")]
        audio = random.choice(file_list)

    if selected_demo_option is not None:
        audio = os.path.join(folder_path, audio)
        st.audio(audio)
        process_audio(audio)


elif selected_option == "Upload file":
    uploaded_file = st.file_uploader("Загрузите wav файл", type=["wav"])
    if uploaded_file:
        st.audio(uploaded_file)
        process_audio(uploaded_file)




elif selected_option == "Record file":
    audio_data = st_audiorec()

    if audio_data is not None:
        with open("audio.wav", "wb") as f:
            f.write(audio_data)
        process_audio("audio.wav")

elif selected_option == "Audio GPT":

    openai_api_key = "your_key"

    st.write("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")

    st.title("💬 Audio Chatbot")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    audio_data = st_audiorec()
    text_data = st.chat_input()

    prompt = None
    if text_data is not None:
        prompt = text_data
    if audio_data is not None:
        with open("prompt.wav", "wb") as f:
            f.write(audio_data)

        recorded_audio, sample_rate = torchaudio.load("prompt.wav", normalize=True)
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        recorded_audio = resampler(recorded_audio)

        if recorded_audio.shape[0] > 1:
            recorded_audio = torch.mean(recorded_audio, dim=0, keepdim=True)

        prompt = transcript_audio(recorded_audio)

    if prompt is not None or audio_data is not None:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        client = OpenAI(api_key=openai_api_key)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                {"role": "user", "content": prompt}
            ]
        )

        st.chat_message("assistant").write(completion.choices[0].message)
