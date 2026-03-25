@echo off

python -m nuitka ^
--mode=onefile ^
--include-package=PIL,numpy,cv2,pyautogui ^
--include-module=main ^
--follow-imports ^
--windows-uac-admin ^
--nofollow-import-to=conftest ^
--output-folder-name=dist ^
.\main.py