# PARL

Implementation of PARL.

## Requirements

```
numpy >= 1.23.0
pandas >= 1.4.3
scipy >= 1.8.1
scikit-learn >= 1.1.1
torch >= 1.10.0
torchtext >= 0.11.0
python >= 3.8
```

## Contents

The project contains 7 folders and 4 files.

1. data (folder): All datasets are in this folder.
2. detect (folder): This folder is used to store the results of detection.
3. log (folder): This folder is used to store the logs.
4. rating_poi (folder): This folder is used to store the poison data.
5. result (folder): This folder is used to store the experimental results.
6. save_weights (folder): This folder is used to store the model weights.
7. util (folder): The folder includes some functional scripts that are needed for running the experiment.
8. parl_detect_svm.py (file): The file is used to obtain the results of detection.
9. parl_gen_poi.py (file): The file is used to generate the poison data.
10. parl_retrain.py (file): The file is used to obtain the attack performance under the white-box settings.
11. parl_retrain_transfer.py (file): The file is used to evaluate the transferability of PARL.

## Run

```
###### Example 1: Poison data generation ######
python parl_gen_poi.py --args xxxx ...

###### Example 2: Attack performance ######
python parl_retrain.py --args xxxx ...

###### Example 3: Transferability ######
python parl_retrain_transfer.py --args xxxx ...

###### Example 4: Detection ######
python parl_detect_svm.py
```

## Citation

```
 @inproceedings{Du2024PARL,
    author = {Linkang Du and Quan Yuan and Min Chen and Mingyang Sun and Peng Cheng and Jiming Chen and Zhikun Zhang},
    title = {{PARL: Poisoning Attacks Against Reinforcement Learning-based Recommender Systems}},
    booktitle = {{AsiaCCS}},
    publisher = {},
    year = {2024},
}
```