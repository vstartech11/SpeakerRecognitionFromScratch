import os

import librosa
import soundfile as sf
from pydub import AudioSegment, silence

import myconfig


def split_by_utterance(input_wav, output_dir, silence_thresh=-40, min_silence_len=500):
    """
    Split a WAV file into utterances based on detected silences (speaker breaks).

    Parameters:
        input_wav (str): Path to the input WAV file.
        output_dir (str): Directory to save the split audio files.
        silence_thresh (int): Silence threshold in dBFS. Default is -40 dBFS.
        min_silence_len (int): Minimum silence length to consider a break in milliseconds. Default is 500 ms.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the input WAV file
    audio = AudioSegment.from_wav(input_wav)

    # Detect non-silent chunks (utterances)
    chunks = silence.split_on_silence(audio,
                                       min_silence_len=min_silence_len,
                                       silence_thresh=silence_thresh,
                                       keep_silence=200)  # Retain a bit of silence in chunks

    # Save each chunk as a separate file with the desired naming convention
    for i, chunk in enumerate(chunks):
        output_file = os.path.join(output_dir, f"G0001_0_S{str(i + 1).zfill(4)}.wav")
        chunk.export(output_file, format="wav")
        print(f"Saved: {output_file}")

        # Extract MFCC features from the saved chunk
        extract_mfcc_features(output_file)

def extract_mfcc_features(audio_file):
    """
    Extract MFCC features from an audio file, shape=(TIME, MFCC).
    """
    waveform, sample_rate = sf.read(audio_file)

    # Convert to mono-channel.
    if len(waveform.shape) == 2:
        waveform = librosa.to_mono(waveform.transpose())

    # Convert to 16kHz.
    if sample_rate != myconfig.SAMPLE_RATE:
        waveform = librosa.resample(y=waveform, orig_sr=sample_rate, target_sr=myconfig.SAMPLE_RATE)

    features = librosa.feature.mfcc(
        y=waveform, sr=myconfig.SAMPLE_RATE, n_mfcc=myconfig.N_MFCC)
    
    print(f"Extracted MFCC features from {audio_file}, shape={features.shape}")

# Example usage
if __name__ == "__main__":
    import myconfig

    # Input WAV file
    input_wav = r"G:\ITK\Project2\Dataset\datasets\G0001\G00001_0_S0001.wav"

    # Output directory
    output_dir = os.path.join(f"{myconfig.DATA_PATH}/datasets/Bintang/G0043")

    # Parameters for silence detection
    silence_thresh = -30  # Silence threshold in dBFS
    min_silence_len = 300  # Minimum silence length in milliseconds

    # Perform the split
    split_by_utterance(input_wav, output_dir, silence_thresh, min_silence_len)