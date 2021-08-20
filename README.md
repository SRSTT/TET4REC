# TET4REC
This repository implements the TET4REC 
# Usage
## Overall
Run main.py with arguments to train  or test you model.
To start, first download ml-1m and ml-20m dataset and put them in the corresponding folder. 
With running main.py, you can choose by entering 1 or 20 for corresponding dataset
After training, you can choose wheter to test with y or n
## TET4REC 
python3 main.py --template train_bert
## hyper-parameter settings
Specific parameter can be adjusted by modifying templates.py
## Requirement
wget==3.2  
tqdm==4.36.1  
numpy==1.16.2  
torch==1.3.0  
tb-nightly==2.1.0a20191121  
pandas==0.25.0  
scipy==1.3.2  
future==0.18.2  
