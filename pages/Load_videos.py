# Environment Configuration
# -----------------------------------------------------------------------
import os
import streamlit as st
import yt_dlp
import whisper
import tempfile
import shutil

# Create directory for storing transcriptions
# -----------------------------------------------------------------------
TRANSCRIPTIONS_FOLDER = "transcriptions"
os.makedirs(TRANSCRIPTIONS_FOLDER, exist_ok=True)

def download_vimeo_audio(url, output_path):
    """
    Download the audio from a Vimeo video.

    This function downloads the best available audio format for the given Vimeo URL,
    converts it to MP3 using FFmpeg, and saves it to the specified output path.

    Parameters
    ----------
    url : str
        The Vimeo video URL.
    output_path : str
        The output template path for saving the audio file.
    """
    options = {
        "format": "bestaudio",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192"
        }]
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])


def transcribe_audio(model, audio_path):
    """
    Transcribe an audio file using the Whisper model.

    Parameters
    ----------
    model : whisper.model
        The loaded Whisper model.
    audio_path : str
        The path to the audio file.

    Returns
    -------
    str
        The transcription text.
    """
    result = model.transcribe(audio_path)
    return result["text"]


def process_urls(model, urls):
    """
    Process a list of Vimeo URLs: download audio, transcribe, save transcription,
    and automatically remove the temporary audio folder after processing.

    Parameters
    ----------
    model : whisper.model
        The loaded Whisper model.
    urls : list of str
        A list of Vimeo video URLs.
    """
    total = len(urls)
    progress_bar = st.progress(0)
    
    # Create a temporary directory for audio files
    with tempfile.TemporaryDirectory() as temp_audio_folder:
        for idx, url in enumerate(urls):
            st.write(f"🔽 Processing: {url}")
            
            # Define output template for the audio file (filename based on video title)
            output_audio_template = os.path.join(temp_audio_folder, "%(title)s.%(ext)s")
            download_vimeo_audio(url, output_audio_template)
            
            # Locate the downloaded MP3 file in the temporary folder
            mp3_files = [f for f in os.listdir(temp_audio_folder) if f.endswith(".mp3")]
            if not mp3_files:
                st.error(f"⚠️ No audio file found for {url}")
                continue

            audio_file = os.path.join(temp_audio_folder, mp3_files[0])
            transcription_text = transcribe_audio(model, audio_file)
            
            # Save the transcription to a text file in the transcriptions folder
            transcription_file_path = os.path.join(
                TRANSCRIPTIONS_FOLDER, f"{os.path.splitext(mp3_files[0])[0]}.txt"
            )
            with open(transcription_file_path, "w", encoding="utf-8") as f:
                f.write(transcription_text)
            
            st.success(f"✅ Transcription saved: {transcription_file_path}")
            st.text_area("Transcription", transcription_text, height=200)
            
            progress_bar.progress((idx + 1) / total)
    # The temporary audio folder is automatically removed here.


def clear_transcriptions():
    """
    Delete all files in the transcriptions folder.

    This function iterates through the 'transcriptions' folder and removes each file.
    If the file is a directory, it is removed recursively.
    """
    if os.path.exists(TRANSCRIPTIONS_FOLDER):
        for filename in os.listdir(TRANSCRIPTIONS_FOLDER):
            file_path = os.path.join(TRANSCRIPTIONS_FOLDER, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                st.error(f"Failed to delete {file_path}. Reason: {e}")
        st.success("All transcription files have been deleted.")
    else:
        st.info("Transcriptions folder does not exist.")


def list_transcriptions():
    """
    List all files in the transcriptions folder.

    This function displays the names of all files in the transcriptions folder.
    """
    if os.path.exists(TRANSCRIPTIONS_FOLDER):
        files = os.listdir(TRANSCRIPTIONS_FOLDER)
        if files:
            st.write("### Transcription Files:")
            for file in files:
                st.write(file)
        else:
            st.info("No transcription files found.")
    else:
        st.info("Transcriptions folder does not exist.")


# Streamlit Application Setup
# -----------------------------------------------------------------------
st.title("Video transcription")

# Session state for model loading
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False

# Button to load the Whisper model and display the URL input menu
if not st.session_state["model_loaded"]:
    if st.button("Load model and show menu"):
        with st.spinner("Loading Whisper base model..."):
            st.session_state["model"] = whisper.load_model("base")
            st.session_state["model_loaded"] = True
        st.success("Model loaded successfully. Click again for entering URLs")
else:
    # Display menu for entering URLs
    urls_input = st.text_area("Enter video URLs (one per line):")
    if st.button("Process URLs"):
        urls = [line.strip() for line in urls_input.splitlines() if line.strip()]
        if urls:
            process_urls(st.session_state["model"], urls)
        else:
            st.error("No valid URLs provided.")

with st.sidebar:
    # Button to clear all transcriptions from the transcriptions folder
    if st.button("Clear transcriptions", type="secondary", help="Deletes all .txt files with transcriptions."):
        clear_transcriptions()

    # Button to list all transcription files
    if st.button("List transcriptions", type="primary", help="Provides with the list of current "):
        list_transcriptions()
