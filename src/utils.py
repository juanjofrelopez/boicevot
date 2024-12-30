import numpy as np

def adapt_audio_for_transcription(audio_data, sample_rate: int = 16000, sample_width: int = 2):
    """
    Adapts raw audio data from get_raw_data() to a format suitable for the transcribe() function.
    
    Args:
        audio_data: Raw PCM audio data from get_raw_data()
        sample_rate: The sample rate of the audio in Hz (default: 16000)
        sample_width: The width of each sample in bytes (default: 2)
        
    Returns:
        numpy.ndarray: Normalized float32 audio array suitable for transcription
    """
    # Convert raw bytes to numpy array based on sample width
    dtype_map = {
        1: np.int8,
        2: np.int16,
        3: np.int32,  # Note: This is an approximation for 24-bit audio
        4: np.int32
    }
    
    if sample_width not in dtype_map:
        raise ValueError(f"Unsupported sample width: {sample_width}")
    
    # Convert bytes to numpy array
    audio_np = np.frombuffer(audio_data, dtype=dtype_map[sample_width])
    
    # If the audio is 24-bit, we need to handle it specially
    if sample_width == 3:
        # Shift the 24-bit values to use the full 32-bit range
        audio_np = audio_np << 8
    
    # Normalize the audio to float32 in the range [-1, 1]
    max_value = float(np.iinfo(dtype_map[sample_width]).max)
    audio_normalized = audio_np.astype(np.float32) / max_value
    
    return audio_normalized
