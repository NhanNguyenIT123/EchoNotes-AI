import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """
    Ensures data folders exist and prints basic diagnostic information.
    """
    print("=" * 60)
    print("[*] EchoNotes AI - Launcher & Environment Check")
    print("=" * 60)
    
    # Path diagnostic
    base_dir = Path(__file__).resolve().parent
    print(f"[*] Base Directory: {base_dir}")
    
    # Check Python version
    print(f"[*] Python Version: {sys.version.split()[0]}")
    
    # Define directories to create
    directories = [
        base_dir / "data" / "raw",
        base_dir / "data" / "processed" / "audio",
        base_dir / "data" / "processed" / "frames",
        base_dir / "data" / "processed" / "transcripts",
        base_dir / "data" / "outputs"
    ]
    
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"[+] Created local workspace directory: {directory.relative_to(base_dir)}")
            
    print("[*] Local folders prepared successfully.")

def run_streamlit_app():
    """
    Launches the Streamlit dashboard app on local port 8501.
    """
    base_dir = Path(__file__).resolve().parent
    ui_script = base_dir / "app" / "ui.py"
    
    if not ui_script.exists():
        print(f"[-] Error: Could not find Streamlit entrypoint file at: {ui_script}")
        sys.exit(1)
        
    print("[*] Launching Streamlit web app...")
    print("[*] Access the dashboard in your browser at http://localhost:8501")
    print("=" * 60)
    
    # Run streamlit run app/ui.py
    cmd = [sys.executable, "-m", "streamlit", "run", str(ui_script)]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[+] EchoNotes AI stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n[-] Error running Streamlit dashboard: {e}")
        print("Please check if streamlit is installed using: pip install -r requirements.txt")

if __name__ == "__main__":
    setup_environment()
    run_streamlit_app()
