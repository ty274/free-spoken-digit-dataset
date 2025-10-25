#!/usr/bin/env python3
"""
Audio file organization utility.
Organizes .wav files hierarchically by digit label and speaker name.
Uses librosa and numpy for audio processing.
"""

import argparse
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import numpy.typing as npt
import librosa
import soundfile as sf


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Organize audio files hierarchically by digit label and speaker name.'
    )
    parser.add_argument(
        '--inputdir',
        type=str,
        required=True,
        help='Directory where audio files are retrieved'
    )
    parser.add_argument(
        '--outputdir',
        type=str,
        required=True,
        help='Output audio directory'
    )
    parser.add_argument(
        '--outputfile',
        type=str,
        required=True,
        help='Output file listing generated audio files (not used yet)'
    )
    parser.add_argument(
        '--ndigits',
        type=int,
        default=30,
        help='Number of audio files to concatenate (default: 30)'
    )
    parser.add_argument(
        '--pause_dur',
        type=float,
        nargs=2,
        default=[5.0, 8.0],
        metavar=('MIN', 'MAX'),
        help='Min and max pause duration in seconds (default: 5.0 8.0)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['repeat', 'no_repeat'],
        default='repeat',
        help='Concatenation mode: repeat (same digit) or no_repeat (different adjacent digits) (default: repeat)'
    )
    return parser.parse_args()


def parse_filename(filename: str) -> tuple[str, str, int] | None:
    """
    Parse a wav filename to extract digit label, speaker name, and index.
    
    Args:
        filename: String in format {digitLabel}_{speakerName}_{index}.wav
        
    Returns:
        tuple: (digitLabel, speakerName, index) or None if parsing fails
    """
    if not filename.endswith('.wav'):
        return None
    
    # Remove .wav extension
    name_without_ext = filename[:-4]
    
    # Split by underscore
    parts = name_without_ext.split('_')
    
    if len(parts) != 3:
        return None
    
    digit_label, speaker_name, index_str = parts
    
    # Convert index to integer for sorting
    try:
        index = int(index_str)
    except ValueError:
        return None
    
    return digit_label, speaker_name, index


def organize_audio_files(input_dir: str) -> dict[str, dict[str, list[str]]]:
    """
    Organize .wav files hierarchically into a dictionary structure.
    
    Args:
        input_dir: Path to directory containing .wav files
        
    Returns:
        dict: Nested dictionary with structure:
              digitLabel -> speakerName -> [list of wav file paths sorted by index]
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    
    # Create nested dictionary structure
    # Store tuples of (index, filepath) temporarily for sorting
    temp_data: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    
    # Retrieve all .wav files
    wav_files = list(input_path.glob('*.wav'))
    
    print(f"Found {len(wav_files)} .wav files in {input_dir}")
    
    # Process each wav file
    skipped = 0
    for wav_file in wav_files:
        parsed = parse_filename(wav_file.name)
        
        if parsed is None:
            print(f"Warning: Skipping file with invalid format: {wav_file.name}")
            skipped += 1
            continue
        
        digit_label, speaker_name, index = parsed
        
        # Store with index for sorting
        temp_data[digit_label][speaker_name].append((index, str(wav_file)))
    
    # Convert to final structure with sorted lists
    data: dict[str, dict[str, list[str]]] = {}
    for digit_label in temp_data:
        data[digit_label] = {}
        for speaker_name in temp_data[digit_label]:
            # Sort by index (first element of tuple) and extract filepath
            sorted_files = sorted(temp_data[digit_label][speaker_name], key=lambda x: x[0])
            data[digit_label][speaker_name] = [filepath for index, filepath in sorted_files]
    
    print(f"Successfully organized {len(wav_files) - skipped} files")
    if skipped > 0:
        print(f"Skipped {skipped} files with invalid format")
    
    return data


def read_audio_file(filepath: str) -> tuple[npt.NDArray[np.float32], int]:
    """
    Read an audio file and return its data and sample rate.
    
    Args:
        filepath: Path to the audio file
        
    Returns:
        tuple: (audio_data, sample_rate) where audio_data is a numpy float array
    """
    audio_data, sample_rate = librosa.load(filepath, sr=None, mono=False)
    return audio_data, sample_rate


def generate_silence(sample_rate: int, duration_seconds: float, num_channels: int = 1) -> npt.NDArray[np.float32]:
    """
    Generate silence for the given duration.
    
    Args:
        sample_rate: Sample rate in Hz
        duration_seconds: Duration of silence in seconds
        num_channels: Number of audio channels (1 for mono, 2 for stereo)
        
    Returns:
        numpy.ndarray: Silent audio data as float array
    """
    num_samples = int(sample_rate * duration_seconds)
    
    if num_channels == 1:
        silence = np.zeros(num_samples, dtype=np.float32)
    else:
        silence = np.zeros((num_channels, num_samples), dtype=np.float32)
    
    return silence


def generate_white_noise(num_samples: int, num_channels: int = 1) -> npt.NDArray[np.float32]:
    """
    Generate white noise (Gaussian).
    
    Args:
        num_samples: Number of samples to generate
        num_channels: Number of audio channels (1 for mono, 2 for stereo)
        
    Returns:
        numpy.ndarray: White noise as float array
    """
    if num_channels == 1:
        white_noise = np.random.randn(num_samples).astype(np.float32)
    else:
        white_noise = np.random.randn(num_channels, num_samples).astype(np.float32)
    
    return white_noise


def add_white_noise(audio: npt.NDArray[np.float32], snr_db: float = 30) -> npt.NDArray[np.float32]:
    """
    Add white noise to audio at specified SNR.
    
    Args:
        audio: Input audio as numpy array (mono or stereo)
        snr_db: Signal-to-noise ratio in decibels
        
    Returns:
        numpy.ndarray: Audio with added white noise
    """
    # Determine number of channels and samples
    if audio.ndim == 1:
        num_channels = 1
        num_samples = len(audio)
    else:
        num_channels = audio.shape[0]
        num_samples = audio.shape[1]
    
    # Generate white noise
    white_noise = generate_white_noise(num_samples, num_channels)
    
    # Calculate signal power
    signal_power = np.mean(audio ** 2)
    
    # Calculate noise power for desired SNR
    # SNR = 10 * log10(signal_power / noise_power)
    # noise_power = signal_power / (10 ^ (SNR/10))
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Scale white noise to desired power
    current_noise_power = np.mean(white_noise ** 2)
    if current_noise_power > 0:
        noise_scale = np.sqrt(noise_power / current_noise_power)
        scaled_noise = white_noise * noise_scale
    else:
        scaled_noise = white_noise
    
    # Add noise to signal
    noisy_audio = audio + scaled_noise
    
    return noisy_audio.astype(np.float32)


def concatenate_audio_files(file_list: list[str], output_path: Path, pause_min: float, pause_max: float) -> None:
    """
    Concatenate audio files with random pauses between them.
    Adds 1 second of silence at the beginning and end.
    
    Args:
        file_list: List of audio file paths to concatenate
        output_path: Output path for the concatenated audio file
        pause_min: Minimum pause duration in seconds
        pause_max: Maximum pause duration in seconds
    """
    if not file_list:
        print(f"Warning: No files to concatenate for {output_path}")
        return
    
    # Read the first file to get audio parameters
    first_audio, first_sr = read_audio_file(file_list[0])
    
    # Determine if mono or stereo
    if first_audio.ndim == 1:
        num_channels = 1
    else:
        num_channels = first_audio.shape[0]
    
    # Start with 1 second of silence
    concatenated_audio: list[npt.NDArray[np.float32]] = [generate_silence(first_sr, 1.0, num_channels)]
    
    # Add the first file's data
    concatenated_audio.append(first_audio)
    
    # Process remaining files
    for i, filepath in enumerate(file_list[1:], start=1):
        # Generate random pause
        pause_duration = random.uniform(pause_min, pause_max)
        silence = generate_silence(first_sr, pause_duration, num_channels)
        concatenated_audio.append(silence)
        
        # Read and append the next audio file
        audio_data, sample_rate = read_audio_file(filepath)
        
        # Verify parameters match
        audio_channels = 1 if audio_data.ndim == 1 else audio_data.shape[0]
        
        if audio_channels != num_channels or sample_rate != first_sr:
            print(f"Warning: Audio parameters mismatch in {filepath}")
            print(f"  Expected: {num_channels} channel(s), {first_sr} Hz")
            print(f"  Got: {audio_channels} channel(s), {sample_rate} Hz")
            continue
        
        concatenated_audio.append(audio_data)
    
    # Add 1 second of silence at the end
    concatenated_audio.append(generate_silence(first_sr, 1.0, num_channels))
    
    # Concatenate all audio segments
    final_audio = np.concatenate(concatenated_audio, axis=-1)
    
    # Add white noise at 30 dB SNR
    final_audio = add_white_noise(final_audio, snr_db=30)
    
    # Write the concatenated audio to output file
    # Transpose if stereo (soundfile expects shape (samples, channels))
    if num_channels > 1:
        final_audio = final_audio.T
    
    sf.write(str(output_path), final_audio, first_sr)
    
    print(f"Created: {output_path.name}")


def get_speaker_digits(data: dict[str, dict[str, list[str]]], speaker_name: str) -> dict[str, list[str]]:
    """
    Get all digit labels and their files for a specific speaker.
    
    Args:
        data: Organized audio file dictionary
        speaker_name: Name of the speaker
        
    Returns:
        dict: Dictionary mapping digit_label -> [list of file paths] for the speaker
    """
    speaker_digits: dict[str, list[str]] = {}
    
    for digit_label in data:
        if speaker_name in data[digit_label]:
            speaker_digits[digit_label] = data[digit_label][speaker_name]
    
    return speaker_digits


def select_no_repeat_sequence(data: dict[str, dict[str, list[str]]], 
                               start_digit: str, 
                               speaker_name: str, 
                               n_digits: int) -> tuple[list[str], list[str]] | None:
    """
    Select a sequence of audio files where neighboring digits are different.
    
    Args:
        data: Organized audio file dictionary
        start_digit: Starting digit label
        speaker_name: Speaker name
        n_digits: Total number of digits to select
        
    Returns:
        tuple: (list of file paths, list of digit labels) or None if not possible
    """
    # Get all available digits for this speaker
    speaker_digits = get_speaker_digits(data, speaker_name)
    
    if start_digit not in speaker_digits:
        return None
    
    # Check if speaker has at least 2 different digits
    if len(speaker_digits) < 2:
        print(f"Warning: Speaker {speaker_name} only has 1 digit type, cannot use no_repeat mode")
        return None
    
    selected_files: list[str] = []
    selected_digits: list[str] = []
    
    # Start with a random file from the start_digit
    current_digit = start_digit
    available_files = speaker_digits[current_digit]
    
    if not available_files:
        return None
    
    # Select random file from start digit
    selected_file = random.choice(available_files)
    selected_files.append(selected_file)
    selected_digits.append(current_digit)
    
    # Select remaining n_digits - 1 files
    for _ in range(n_digits - 1):
        # Get available digits (excluding current digit)
        available_digits = [d for d in speaker_digits.keys() if d != current_digit and speaker_digits[d]]
        
        if not available_digits:
            print(f"Warning: Cannot find different digit for speaker {speaker_name} after {current_digit}")
            return None
        
        # Randomly choose next digit
        next_digit = random.choice(available_digits)
        
        # Randomly choose a file from that digit
        next_file = random.choice(speaker_digits[next_digit])
        
        selected_files.append(next_file)
        selected_digits.append(next_digit)
        
        current_digit = next_digit
    
    return selected_files, selected_digits


def process_and_concatenate(data: dict[str, dict[str, list[str]]], output_dir: str, 
                           output_file: str, n_digits: int, pause_min: float, 
                           pause_max: float, mode: str) -> None:
    """
    Process the organized data and create concatenated audio files.
    
    Args:
        data: Organized audio file dictionary
        output_dir: Output directory path
        output_file: Path to output TSV file
        n_digits: Number of audio files to concatenate
        pause_min: Minimum pause duration in seconds
        pause_max: Maximum pause duration in seconds
        mode: Concatenation mode ('repeat' or 'no_repeat')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create directory for output TSV file if it doesn't exist
    output_file_path = Path(output_file)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating concatenated audio files in: {output_dir}")
    print(f"Mode: {mode}")
    print(f"Concatenating {n_digits} files per digit-speaker pair")
    print(f"Random pause duration: {pause_min} - {pause_max} seconds")
    print(f"Adding 1 second of silence at beginning and end")
    print(f"Adding white noise at 30 dB SNR\n")
    
    files_created = 0
    files_skipped = 0
    tsv_lines: list[str] = []
    tsv_lines.append("file_path\ttarget")

    if mode == 'repeat':
        # Original repeat mode
        for digit_label in sorted(data.keys()):
            for speaker_name in sorted(data[digit_label].keys()):
                file_list = data[digit_label][speaker_name]
                
                # Check if we have enough files
                if len(file_list) < n_digits:
                    print(f"Warning: {digit_label}_{speaker_name} has only "
                          f"{len(file_list)} files (need {n_digits}), skipping")
                    files_skipped += 1
                    continue
                
                # Take the first N files
                files_to_concat = file_list[:n_digits]
                
                # Create output filename
                output_filename = f"{digit_label}x{n_digits}_{speaker_name}.wav"
                output_filepath = output_path / output_filename
                
                # Concatenate the files
                concatenate_audio_files(files_to_concat, output_filepath, 
                                       pause_min, pause_max)
                files_created += 1
                
                # Generate transcription: repeat digit label n_digits times
                transcription = ' '.join([digit_label] * n_digits)
                
                # Add to TSV lines (store absolute path)
                tsv_lines.append(f"{output_filepath.absolute()}\t{transcription}")
    
    elif mode == 'no_repeat':
        # New no_repeat mode
        for start_digit in sorted(data.keys()):
            for speaker_name in sorted(data[start_digit].keys()):
                # Select sequence with no repeating adjacent digits
                result = select_no_repeat_sequence(data, start_digit, speaker_name, n_digits)
                
                if result is None:
                    print(f"Warning: Cannot create no_repeat sequence for {start_digit}_{speaker_name}, skipping")
                    files_skipped += 1
                    continue
                
                files_to_concat, digit_sequence = result
                
                # Create output filename: concatenate all digits + speaker name
                digit_string = ''.join(digit_sequence)
                output_filename = f"{digit_string}_{speaker_name}.wav"
                output_filepath = output_path / output_filename
                
                # Concatenate the files
                concatenate_audio_files(files_to_concat, output_filepath, 
                                       pause_min, pause_max)
                files_created += 1
                
                # Generate transcription: space-separated digit sequence
                transcription = ' '.join(digit_sequence)
                
                # Add to TSV lines (store absolute path)
                tsv_lines.append(f"{output_filepath.absolute()}\t{transcription}")
    
    # Write TSV file
    with open(output_file_path, 'w') as f:
        for line in tsv_lines:
            f.write(line + '\n')
    
    print(f"\n{'=' * 60}")
    print(f"Concatenation complete!")
    print(f"Files created: {files_created}")
    print(f"Files skipped (insufficient audio): {files_skipped}")
    print(f"TSV file written to: {output_file_path.absolute()}")
    print(f"{'=' * 60}")


def print_data_summary(data: dict[str, dict[str, list[str]]]) -> None:
    """Print a summary of the organized data structure."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    for digit_label in sorted(data.keys()):
        print(f"\nDigit Label: {digit_label}")
        for speaker_name in sorted(data[digit_label].keys()):
            file_count = len(data[digit_label][speaker_name])
            print(f"  Speaker: {speaker_name} - {file_count} file(s)")
            # Show first 3 files as examples
            for i, filepath in enumerate(data[digit_label][speaker_name][:3]):
                print(f"    - {Path(filepath).name}")
            if file_count > 3:
                print(f"    ... and {file_count - 3} more")


def main() -> None:
    """Main function."""
    # Set random seed for reproducibility
    np.random.seed(0)
    random.seed(0)
    
    args = parse_arguments()
    
    pause_min, pause_max = args.pause_dur
    
    print(f"Input directory: {args.inputdir}")
    print(f"Output directory: {args.outputdir}")
    print(f"Output TSV file: {args.outputfile}")
    print(f"Number of digits to concatenate: {args.ndigits}")
    print(f"Pause duration range: {pause_min} - {pause_max} seconds")
    print(f"Mode: {args.mode}")
    print()
    
    # Organize audio files
    data = organize_audio_files(args.inputdir)
    
    # Print summary
    print_data_summary(data)
    
    print("\n" + "=" * 60)
    print(f"Total digit labels: {len(data)}")
    total_speakers = sum(len(speakers) for speakers in data.values())
    print(f"Total unique speakers: {total_speakers}")
    total_files = sum(
        len(files) 
        for speakers in data.values() 
        for files in speakers.values()
    )
    print(f"Total files organized: {total_files}")
    print("=" * 60)
    
    # Process and concatenate audio files
    process_and_concatenate(data, args.outputdir, args.outputfile, args.ndigits, 
                           pause_min, pause_max, args.mode)


if __name__ == '__main__':
    main()