# Text Classification with HuggingFace🤗 BERT

这里包含本次作业的所有相关文件，你可以基于此仓库进行训练或预测。



## Introduction of files

```shell
│  22Spring_ContemporaryAI_P1_Text_Classification.pdf	# 实验报告
│  README.md											# 本文件
│  requirements.txt										# 环境配置文件
│  submit_sample.txt									# 预测结果
│  train_model.py										# 运行以训练模型
│  use_model.py											# 运行以预测数据
│
├─best_model											# 已训练好的模型和参数
│      config.json
│      pytorch_model.bin
│      training_args.bin
│
└─dataset												# 数据
        test.txt										# 测试数据
        train_data.txt									# 训练数据
```



## Environment setup

环境依赖已经列在 requirements.txt 中了，使用

```shell
pip install -r requirements.txt
```

安装。

注意：如果使用 GPU，请安装对应版本的 PyTorch 和 CUDA。



## Predict

使用

```shell
python use_model.py
```

预测 test.txt 中的数据。

预测结果将在当前文件夹中以 submit_sample.txt 的格式保存。



## Train

使用

```shell
python train_model.py
```

训练模型。

为了避免训练结果覆盖掉我上传的 best_model，训练的模型将保存在当前路径下的 model_train 文件夹中。



## Contact information

Please contact Hao-Ran Yang (`younghojann@gmail.com`).