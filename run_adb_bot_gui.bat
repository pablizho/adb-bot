@echo off
setlocal enableextensions
REM -------------------------------------------------------------
REM ADB Bot GUI launcher (Windows)
REM -------------------------------------------------------------

REM 1) Встаём в папку скрипта (учитывает пробелы и OneDrive)
pushd "%~dp0"

REM 2) Читаемый вывод и русская кодировка (не критично)
chcp 65001 >NUL 2>&1

echo ==============================
echo Запуск GUI ADB Bot
echo Папка проекта: %CD%
echo ==============================
echo.

REM 3) Убедимся, что есть Python
where py >NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
  where python >NUL 2>&1 || (
    echo [ОШИБКА] Python не найден. Установи Python 3.11+ ^(x64^) и перезапусти.
    goto :PAUSE_AND_EXIT
  )
)

REM 4) Создаём venv, если его нет
if not exist ".venv\Scripts\python.exe" (
  echo Создаю виртуальное окружение .venv ...
  if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
    "%LocalAppData%\Programs\Python\Python313\python.exe" -m venv .venv || (
      echo [ОШИБКА] Не удалось создать .venv
      goto :PAUSE_AND_EXIT
    )
  ) else (
    REM Падаем обратно на py/python из PATH
    py -3 -m venv .venv 2>nul || python -m venv .venv || (
      echo [ОШИБКА] Не удалось создать .venv
      goto :PAUSE_AND_EXIT
    )
  )
)

REM 5) Активируем venv
call ".venv\Scripts\activate.bat"

REM 6) Фикс Tk/Tcl для Python 3.13 (как ты делал вручную)
REM    Если у тебя другая версия — поправь PY313_HOME ниже.
set "PY313_HOME=%LocalAppData%\Programs\Python\Python313"
if exist "%PY313_HOME%\tcl\tcl8.6" (
  set "TCL_LIBRARY=%PY313_HOME%\tcl\tcl8.6"
  set "TK_LIBRARY=%PY313_HOME%\tcl\tk8.6"
)

REM 7) Обновим pip и зависимости
python -m pip --disable-pip-version-check -q install --upgrade pip
if exist requirements.txt (
  echo Устанавливаю зависимости из requirements.txt ...
  python -m pip -q install -r requirements.txt
) else (
  echo Устанавливаю необходимые пакеты ...
  python -m pip -q install opencv-python numpy pyyaml
)

REM 8) adb: если не найден в PATH, попробуем локальный platform-tools
where adb >NUL 2>&1
if errorlevel 1 (
  if exist "%CD%\platform-tools\adb.exe" (
    set "PATH=%CD%\platform-tools;%PATH%"
  ) else (
    echo [ВНИМАНИЕ] adb не найден в PATH. Установи Android Platform Tools
    echo            или положи их в папку: %CD%\platform-tools
  )
)

echo.
echo Запускаю GUI...
echo (окно не закроется само — по окончании будет пауза)
echo.

REM 9) Запускаем приложение
python "adb_bot_gui.py"
set "RC=%ERRORLEVEL%"
echo.
echo Работа программы завершена. Код выхода: %RC%
echo.

:PAUSE_AND_EXIT
echo Нажми любую клавишу, чтобы закрыть окно...
pause >NUL
popd
endlocal
