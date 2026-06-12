import os

UPLOAD_FOLDER = "uploaded_docs"


def ensure_upload_folder():
    """Create upload folder if it doesn't exist."""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def save_uploaded_file(file_bytes: bytes, filename: str) -> str:
    """Save uploaded file bytes to disk and return the file path."""
    ensure_upload_folder()
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    return file_path