# Dissecting-CoT
This repository contains the source code of our paper. Part of the code is borrowed from [in-context-learning](https://github.com/dtsip/in-context-learning).

**Dissecting Chain-of-Thought: Compositionality through In-Context Filtering and Learning**<br />
Yingcong Li, Kartik Sreenivasan, Angeliki Giannou, Dimitris Papailiopoulos, Samet Oymak<br />
Paper: [https://arxiv.org/pdf/2301.07067.pdf](https://arxiv.org/pdf/2305.18869.pdf)


## Start up
Follow [in-context-learning](https://github.com/dtsip/in-context-learning)'s instruction by runing
```
conda env create -f environment.yml
conda activate in-context-learning
```

## Training
1. Enter into directory ```src```
2. Run ```train.py``` based on the choosen config file: ```python train.py --config conf/[config_file].yaml```

## Test
1. Enter into directory ```src```
2. Refer to the ```eval.ipynb``` file
 
## Contact
If you have any question, please contact **Yingcong Li** (<yli692@ucr.edu>)


