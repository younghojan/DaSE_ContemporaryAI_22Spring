# MFSAM

This is the official repository of Lab 5 of DaSE course *Contemporary AI*.

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:

- torch==1.9.0

- Pillow==9.1.1

- scikit-learn==1.1.1

- torchvision==0.10.0

- transformers==4.19.2

- tqdm==4.64.0

- scipy==1.8.1

- numpy==1.22.4

You can simply run

```shell
pip install -r requirements.txt
```

## Repository structure

We select some important files for detailed description.

```shellsession
│  config.py    # Parse the runtime parameters
│  data_process.ipynb    # preprocess datas
│  README.md
│  requirements.txt
│  run.py    # the main code
│  utils.py    # helper functions
│
├─dataset
│  │  dev.json
│  │  test.json
│  │  test_without_label.txt
│  │  train.json
│  │  train.txt
│  │
│  └─data    # the folder that contains data examples
│
└─models
    │  int_fusion_model.py    # multimodal model
    │  txt_model.py    # text modl
    │  vis_model.py    # image model
```

## Train and test

1. You can train the model by this script:
   
   ```shell
   python run.py --output_dir output/ --train_filepath dataset/train.json --dev_filepath dataset/dev.json --test_filepath dataset/test.json --do_train
   ```

2. You can train the model by this script:
   
   ```shell
   python run.py --output_dir output/ --train_filepath dataset/train.json --dev_filepath dataset/dev.json --test_filepath dataset/test.json --do_test --store_preds
   ```

Remember to check arguments' implications in config.py or run.py if you do not know what an arguent means, or you might make some mistakes.
