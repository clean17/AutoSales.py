@echo off
chcp 65001 >nul

REM 1) 가상환경 활성화 (반드시 call 사용)
call "C:\my-project\AutoSales.py\venv\Scripts\activate.bat"

@REM python 5_find_low_point_test.py
python 7_find_low_point_processPool.py

REM 2) 32비트 환경이 적용된 cmd 창을 계속 열어두기
cmd /k