@echo off
echo ============================================================
echo  Traffic Police CV System - Environment Setup
echo ============================================================
echo.
echo [1/3] Upgrading pip...
C:\Windows\py.exe -3.11 -m pip install --upgrade pip

echo.
echo [2/3] Installing PyTorch 2.3.1 with CUDA 12.1 wheels...
C:\Windows\py.exe -3.11 -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

echo.
echo [3/3] Installing remaining dependencies...
C:\Windows\py.exe -3.11 -m pip install opencv-python==4.10.0.84 Pillow==10.4.0 numpy==1.26.4 scipy==1.14.0 scikit-learn==1.5.1 matplotlib==3.9.1 tqdm==4.66.4 fastapi==0.111.1 "uvicorn[standard]==0.30.3" python-multipart==0.0.9 aiofiles==24.1.0 pandas==2.2.2

echo.
echo ============================================================
echo  Verifying installation...
echo ============================================================
C:\Windows\py.exe -3.11 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

echo.
echo Done! Run: C:\Windows\py.exe -3.11 part1_foundations\dataset.py
pause
