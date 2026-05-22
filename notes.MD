
 3. Goal of DL model would be to classify performance (correct vs incorrect) based on physiological measures


 ### FOR NEXT TIME
 1. Model B - make sure the CLIP + EEG works
 2. Robust enough for many subjects (test) 

### Usage

How to run the streamlit: .venv/bin/streamlit run src/dashboard.py

cd /Users/madhav/PSY197B
streamlit run src/vision/stream_annotator.py --server.port 8502

* use branches for features, always do
git status
git add -p          # or git add <files>
git commit -m "Describe the change"
git push

## Running the App

### Windows (lab machine)
1. First-time setup: double-click `setup_windows.bat` (creates `.venv` and installs packages)
2. Launch dashboard: double-click `run.bat`
3. Launch annotator: `run.bat annotator`
4. If you get import errors, run: `.venv\Scripts\python.exe -m pip install -r requirements.txt`

### Mac / Linux (dev machine)
1. First-time setup: `bash setup_mac.sh`
2. Launch dashboard: `make run`
3. Launch annotator: `make annotator`

### Installing packages
Never use bare `pip install` — packages will go to system Python instead of the venv.

**Windows:** `.venv\Scripts\python.exe -m pip install <package>` (or use `pip.bat install <package>`)
**Mac:** `.venv/bin/pip install <package>`

### Verifying a package is in the venv
**Windows:** `.venv\Scripts\python.exe -c "import <package>; print(<package>.__file__)"`
**Mac:** `.venv/bin/python3 -c "import <package>; print(<package>.__file__)"`
