# -*- coding: utf-8 -*-
import os
import shutil
import zipfile
from pathlib import Path

def main():
    zip_path = Path(r"C:\Users\as\Downloads\whisper-vinglish-ct2 (1).zip")
    dest_dir = Path(r"D:\GITHUB\EchoNotes-AI\data\whisper-vinglish-model\whisper-vinglish-ct2")
    temp_extract = Path(r"D:\GITHUB\EchoNotes-AI\data\whisper-vinglish-model\temp_extract")
    
    print(f"[OK] Starting to process zip file: {zip_path}")
    
    if not zip_path.exists():
        print(f"[Error] Zip file not found at: {zip_path}")
        return
        
    # 1. Unzip to temporary folder
    if temp_extract.exists():
        shutil.rmtree(temp_extract)
    temp_extract.mkdir(parents=True, exist_ok=True)
    
    print("[*] Unzipping new model...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract)
        
    # 2. Find subfolder containing the model files (since Colab zipped the parent folder)
    model_source_dir = temp_extract
    subdirs = [x for x in temp_extract.iterdir() if x.is_dir()]
    if subdirs:
        model_source_dir = subdirs[0]
        
    print(f"[OK] Unzipped model files location: {model_source_dir}")
    
    # Check if model.bin exists
    if not (model_source_dir / "model.bin").exists():
        print("[Error] model.bin not found in unzipped folder!")
        shutil.rmtree(temp_extract)
        return
        
    # 3. Clean and replace the old model folder
    if dest_dir.exists():
        print("[*] Cleaning up old model folder...")
        shutil.rmtree(dest_dir)
        
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Move all files to destination directory
    print("[*] Copying new model files to active directory...")
    for item in model_source_dir.glob("*"):
        shutil.move(str(item), str(dest_dir / item.name))
        
    # 5. Clean up temporary extraction directory
    shutil.rmtree(temp_extract)
    print("[SUCCESS] New model has been successfully updated in the system!")

if __name__ == "__main__":
    main()
