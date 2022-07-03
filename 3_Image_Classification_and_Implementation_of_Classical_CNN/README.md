# Image Classification Using Classical CNNs

这里包含本次作业的所有相关文件，你可以基于此仓库进行训练和预测。



## Introduction of files

```shell
│  22Spring_ContemporaryAI_P3_Image_Classification_and_Implementation_of_Classical_CNN.pdf	# 实验报告
│  README.md				# 本文件
│
└─torch_code				# 模型代码
        AlexNet				# AlexNet torch 模型
        AlexNet.py
        DenseNet			# DenseNet torch 模型
        DenseNet.py
        LeNet				# LeNet torch 模型
        LeNet.py
        main.py
        ResNet				# ResNet torch 模型
        ResNet.py
        VGGNet				# VGGNet torch 模型
        VGGNet.py
```



## Train and predict

通过运行 main.py 可以进行模型的训练，main.py 为训练提供了 4 个可选参数：

- `--model`: 配置模型
- `--lr`: 配置学习率
- `--batch_size`: 配置 batchsize
- `--epoch`: 配置 epoch 数

如，你可以使用下面的命令来训练 LeNet 并且利用你训练的模型在测试集上进行预测：

```shell
python main.py --model LeNet --lr 0.004 --batch_size 128 --epoch 15
```

另外，你可以使用已经训练好的模型，在你的代码中编写 `torch.load()` 加载它们。



## Contact information

Please contact Hao-Ran Yang (`younghojann@gmail.com`).