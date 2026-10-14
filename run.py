"""
One-Click Launch Script for FraudGuard AI
Usage: python run.py
Automatically verifies models, starts the server, and opens your browser.
"""

import os
import sys
import time
import threading
import webbrowser
import subprocess

def main():
    print("====================================================")
    print("💳 FraudGuard AI • Deep Neural Network")
    print("Launching complete project...")
    print("====================================================")

    # 1. Ensure pre-trained model exists
    if not os.path.exists("models/best_dnn_model.pth"):
        print("[INFO] Model weights missing. Running initial quick training...")
        subprocess.run([sys.executable, "-m", "src.train", "--epochs", "6"])

    port = 8502
    url = f"http://localhost:{port}"

    # 2. Automatically open the browser
    def launch_browser():
        time.sleep(2.5)
        print(f"[INFO] Opening browser at: {url}")
        webbrowser.open(url)

    threading.Thread(target=launch_browser, daemon=True).start()

    # 3. Start Streamlit server
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        f"--server.port={port}",
        "--server.headless=true"
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[INFO] Application stopped.")

if __name__ == "__main__":
    main()
