# Voice_Isolation
Final project for MIT's Deep Learning (6.7960) Class. Built a deep-learning pipeline for voice isolation. 

BLOG: https://MarcusBluestone.github.io/Voice_Isolation/

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
cd src
python -m data
```

To run the training, set any .json files in `params` folder. Then run:
```
cd src
python -m train
```

