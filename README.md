# Gesture-Based Media Controls

This project offers an easy way to control your music using your hand and webcam! Simply install the python package and run the server and you can control any media with just your hand!

## Getting started
```bash
git clone https://github.com/nander100/EE146-project.git
cd EE146-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python project.py 
```

## Known issues
1. If the program is always running, if you go up to scratch your face or a hand walks past beyond the camera, then the controls will trigger. This could be annoying for the user and should be fixed in a later revision.

2. If the user wants to skip a bunch of songs quickily, then they will need to first move their hand out of the fov of the camera since moving it back to the original position will return to the previous song.
