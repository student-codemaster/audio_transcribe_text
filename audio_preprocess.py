import librosa
import soundfile as sf
import noisereduce as nr
import os

def preprocess_audio(path):
    """Preprocess audio: load, trim, normalize, and denoise.
    
    Args:
        path: Path to audio file
    
    Returns:
        Path to cleaned audio file
    """
    try:
        # Validate input
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        
        # Load audio with 16kHz sample rate
        audio, sr = librosa.load(path, sr=16000)
        
        if len(audio) == 0:
            raise ValueError("Audio file is empty")
        
        # Trim silence
        trimmed, _ = librosa.effects.trim(audio)
        
        # Normalize
        normalized = librosa.util.normalize(trimmed)
        
        # Denoise
        cleaned = nr.reduce_noise(y=normalized, sr=sr)
        
        # Create output directory if needed
        output_dir = "uploads"
        os.makedirs(output_dir, exist_ok=True)
        
        # Output path with .wav extension
        output = os.path.join(output_dir, "cleaned_audio.wav")
        
        # Write cleaned audio
        sf.write(output, cleaned, sr)
        
        print(f" Audio preprocessed: {output}")
        return output
    
    except Exception as e:
        print(f" Preprocessing error: {str(e)}")
        raise