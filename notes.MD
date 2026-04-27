
 3. Goal of DL model would be to classify performance (correct vs incorrect) based on physiological measures


 ### FOR NEXT TIME
 1. Model B - make sure the CLIP + EEG works
 2. Robust enough for many subjects (test) 

### Usage

How to run the streamlit: .venv/bin/streamlit run src/dashboard.py

* use branches for features, always do
git status
git add -p          # or git add <files>
git commit -m "Describe the change"
git push


#### How CV works? (CLIP Model)

* CLIP is an NN by OpenAI
* we cluster images based on similarity and map into a trial
* helps us avoid hand-labeling images
* uses natural langauge 


4/24/26 meeting notes
- time the eyetracking/vision to triggers
- clean up data (get rid of uneccesary files)
- multi-subject
- look at head movements, imu

Computer Vision:
- Compare ResNet with CLIP 
- Setup labeling infra for other people

EEG:
- GEDAI - baseline correcting? kicking out things? how is this working?
- walking outdoors slack messege for metrics

EEGNet:
- Meet with Jordan for classifying things?
- EEGNet sit vs walk condition
    -why 260 trails, what is the embedding explorer?
    - sitting vs walking
    - go vs no-go
    - how is training structured 


# Questions:

-sj04 walking conditions missing 3d_eye_states, unattend_sit gaze positions are empty