import os
import librosa
import pandas as pd

def analyze_dataset(audio_folder):

    data = []

    for file in os.listdir(audio_folder):

        if file.endswith(".wav"):

            path = os.path.join(audio_folder, file)

            audio, sr = librosa.load(path, sr=None)

            duration = librosa.get_duration(y=audio, sr=sr)

            data.append({
                "file": file,
                "sample_rate": sr,
                "duration_sec": duration
            })

    df = pd.DataFrame(data)

    return df