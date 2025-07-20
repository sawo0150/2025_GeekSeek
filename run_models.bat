@echo off
setlocal

:: 가상 환경 활성화 (현재 경로에 'my_gpu_env'가 있다고 가정)
call "C:\Users\user\Videos\.fmri\2025_GeekSeek\my_gpu_env\Scripts\activate.bat"

:: 각 모델에 대해 순차적으로 main.py 실행
echo Running featurevarnet_sh_w
python main.py --config-name=train_me_varnet_Ver2 model=featurevarnet_sh_w
echo.

echo Running fivarnet
python main.py --config-name=train_me_varnet_Ver2 model=fivarnet
echo.

echo Running ifvarnet
python main.py --config-name=train_me_varnet_Ver2 model=ifvarnet
echo.

echo All models finished.
pause