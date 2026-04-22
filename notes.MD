
### Overall Plan
 1. Include YAML + Streamlit     
 2. Label world video based on eye-tracking 
    a.  downsample world video if necessary
 3. Goal of DL model would be to classify performance (correct vs incorrect) based on physiological measures



 ### FOR NEXT TIME
 1. Vision pipeline accuracy (low acc, categories seem wrong, better summary stats, categories..etc)
 2. Have a proper YAML + Streamlit to play around with everything
 3. Robust enough for many subjects (test) 


How to run the streamlit: .venv/bin/streamlit run src/dashboard.py




# How CV works? (CLIP Model)

* CLIP is an NN by OpenAI
* we cluster images based on similarity and map into a trial
* helps us avoid hand-labeling images
* uses natural langauge 

Mtg Fixes
* ICA (actually kicking out eye blinks??, looks like it isn't)
* Check back matlab code 
* GEDAI denoinsing
* X,Y ET coords graph
    * smoothing function
    * better graph
    * euclidian distance (do they move their eyes more?)


# Everyday git
* use branches for features, always do
git status
git add -p          # or git add <files>
git commit -m "Describe the change"
git push

## QUESTIONS/KEEP IN MIND
* subjects 1-4 have hard-coded trail alignment issues, what about other ones?

*Understand + double check ML implementation, questions, looking at the right things
*Streamlit tab with ML