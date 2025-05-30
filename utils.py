import os

ALLOWED_IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}
ALLOWED_AUDIO_EXT = {'.mp3', '.wav', '.aac', '.flac'}
ALLOWED_VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv'}
ALLOWED_TEXT_EXT = {'.txt', '.csv'}
ALLOWED_TEXT_EXT = {'.txt', '.csv', '.log'}

def validate_file(filename, allowed_extensions):
    return os.path.splitext(filename)[1].lower() in allowed_extensions