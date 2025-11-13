# Voice_Isolation
Final project for MIT's Deep Learning (6.7960) Class. Built a deep-learning pipeline for voice isolation. 

Working Outline: https://docs.google.com/document/d/1Z5bNR8pYPRpAQ_cMwrd0k4K78t8TmwluppZDUIWA5z8/edit?usp=sharing

Download Enviornment:
```
conda env create -f environment.yaml
conda activate voice_isolate_env
```

To re-upload environment if u made changes to it:
```
conda env export > conda_env.yaml
```

To experiment w/ dataset:
```
python -m src.data
```

Run with:
```
python -m src.train
```

