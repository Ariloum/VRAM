@echo off
pyinstaller --noconsole --icon=app.ico --onefile VRAM.py 
pause

..mkdir ..\VRAM-exe

move /y dist\VRAM.exe VRAM.exe
for /f %%a IN ('dir "build\vram" /b') do move "build\vram\%%a" ".\"
move /y build\vram\ exe
pause