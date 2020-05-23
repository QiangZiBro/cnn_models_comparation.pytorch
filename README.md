# CNN 经典模型比较

本项目希望学习以下几点：

- 使用pytorch学习经典CNN网络
- 对不同CNN网络性能进行进行比较
- 使深度学习代码工程化，项目模板参考了[pytorch-template](https://github.com/victoresque/pytorch-template)，用好这些模板会很方便做实验，我对模板进行了部分修改。

## 零、如何运行 

Detailed docs is coming...:beers:

- 准备数据

数据集放在`data/`目录下。通常，我们不建议直接将数据集放在项目里，而是放在一个共有目录下，比如 `~/data`，再使用软链接：

```bash
cd data
ln -s ~/data/cifar-10-batches-py cifar-10-batches-py
cd ..
```

- 运行所有网络，训练的模型和日志储存在`saved/`文件夹里

```bash
# 训练所有网络
bash run.sh
```

- 用训练好的模型对测试集进行评估

```bash
python test.py --resume path-to-checkpoint.pth
```

## 一、操作环境

- OS:Ubuntu16
- GPU:2080Ti
- Cuda:10.0
- Cudnn:7

## 二、目前使用的数据集

- [x] Cifar10
- [ ] Cifar100
- [ ] MNIST

...

## 三、目前使用的模型

- [x] LeNet
- [x] AlexNet
- [x] NiN
- [x] GoogLeNet
- [x] Batch Normalization using LeNet
- [x] ResNet
- [ ] DenseNet
- [ ] CapsNet
- [ ] AdderNet

...

## 四、实验结果

### Cifar10

- early stop:10
- Learning rate:0.001

| 模型                            | 训练准确率 | 测试准确率 |
| ------------------------------- | ---------- | ---------- |
| AlexNet                         |            |            |
| LeNet                           |            |            |
| NiN                             |            |            |
| GoogLeNet                       |            |            |
| Batch Normalization using LeNet |            |            |
| ResNet                          |            |            |

