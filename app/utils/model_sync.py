# -*- coding: utf-8 -*-
import os
import re
import gc
import shutil
import zipfile
import requests
from pathlib import Path

def download_file_from_google_drive(share_url: str, destination: Path) -> bool:
    """
    Downloads a large file from Google Drive via sharing link, 
    bypassing the virus scan warning confirmation automatically.
    """
    # Extract file ID using regex
    file_id = None
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', share_url)
    if match:
        file_id = match.group(1)
    else:
        match2 = re.search(r'id=([a-zA-Z0-9_-]+)', share_url)
        if match2:
            file_id = match2.group(1)
            
    if not file_id:
        raise ValueError("Không thể trích xuất File ID từ đường dẫn Google Drive của bạn! Hãy đảm bảo sử dụng đường dẫn chia sẻ chuẩn.")
        
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    print(f"[Model Sync] Initiating Google Drive download for file ID: {file_id}")
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Check for warning confirmation token
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
            
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
        
    CHUNK_SIZE = 1024 * 1024  # 1MB chunk size for fast download
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                
    return destination.exists()

def extract_and_install_zip(zip_path: Path, dest_dir: Path) -> bool:
    """
    Unzips and installs model files from zip to target destination, 
    cleaning up old files and temp directories.
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    temp_extract = dest_dir.parent / "temp_extract_sync"
    
    if not zip_path.exists():
        return False
        
    # 1. Unzip to temporary folder
    if temp_extract.exists():
        shutil.rmtree(temp_extract)
    temp_extract.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract)
        
    # 2. Find subfolder containing the model files (since Colab zipped the parent folder)
    model_source_dir = temp_extract
    subdirs = [x for x in temp_extract.iterdir() if x.is_dir()]
    if subdirs:
        model_source_dir = subdirs[0]
        
    # Check if model.bin exists
    if not (model_source_dir / "model.bin").exists():
        if temp_extract.exists():
            shutil.rmtree(temp_extract)
        raise ValueError("File ZIP không chứa tệp mô hình model.bin hợp lệ!")
        
    # 3. Clean and replace the old model folder
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Move all files to destination directory
    for item in model_source_dir.glob("*"):
        shutil.move(str(item), str(dest_dir / item.name))
        
    # 5. Clean up temporary directory
    shutil.rmtree(temp_extract)
    gc.collect()
    return True

def auto_detect_google_drive_paths() -> list:
    """
    Scans common Google Drive Desktop paths on Windows to auto-suggest
    valid local synced file paths to the user.
    """
    username = os.getlogin() if hasattr(os, 'getlogin') else 'as'
    potential_paths = [
        r"G:\My Drive\whisper-vinglish-ct2.zip",
        r"G:\My Drive\EchoNotes\whisper-vinglish-ct2.zip",
        f"C:\\Users\\{username}\\Google Drive\\My Drive\\whisper-vinglish-ct2.zip",
        f"C:\\Users\\{username}\\OneDrive\\whisper-vinglish-ct2.zip"
    ]
    
    # Filter only paths that exist
    return [p for p in potential_paths if Path(p).exists()]
