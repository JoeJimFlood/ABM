:: Creating virtual Python environment
set PYTHON_ENV=C:\ProgramData\Anaconda3
call %PYTHON_ENV%\Scripts\activate.bat %PYTHON_ENV%
call conda env list | find /i "viz"
if not errorlevel 1 (
	call conda env update --name viz --file %WORKING_DIR%environment.yml
	call activate viz
) else (
	call conda env create -f %WORKING_DIR%environment.yml
)
call activate viz

call python setup.py

call python data_pipeliner\run.py

call python copy_to_combine.py

call python combine\combine.py

call conda deactivate viz

pause