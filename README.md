# 利用keras实现的SRAN超分辨率重建网络(WGAN)



## 前言
  深度学习小白一枚，刚入门，看到这个有趣的论文，便忍不住复现了一波，期间学到了很多东西，也踩了很多坑，代码或有不周，还请见谅
  
ps:网络结构亲测可用，若有问题，还请发issue

网络结构与论文中相同利用了残差结构，见下图：

<img src="https://github.com/zjq1996518/srgan-with-wgan/blob/master/image/model.jpeg?raw=true"></img>

ps：这个网络很有意思，感觉对训练集做一些处理是不是可以用来美颜，生成化妆之类的，当然也只是猜测，后续也想做一些实验

## 特点：
* 利用暴力的wgan的方式训练模型，判别器最后一层不用sigmoid激活，使用推土机距离(Wasserstein)，迭代完后，将权重裁剪为-0.01 - 0.01 之间（本来想用wgan-gp，但遇到了点问题，后续会改进）
* 训练集使用IMAGENET-2012中的验证集图片，将原图片裁剪为1/2作为生成器输入，输出为原图片(感觉可以在原图片上加一些模糊处理，后续会尝试一下)
* 整个网络采用keras实现，中文注释

## 结果：

#### 第0次迭代
<img src="https://github.com/zjq1996518/srgan-with-wgan/blob/master/image/0_0.jpg?raw=true"></img>

#### 第2000次迭代
<img src="https://github.com/zjq1996518/srgan-with-wgan/blob/master/image/0_2000.jpg?raw=true"></img>

#### 第4000次迭代
<img src="https://github.com/zjq1996518/srgan-with-wgan/blob/master/image/0_4000.jpg?raw=true"></img>

#### 第50000次迭代
<img src="https://github.com/zjq1996518/srgan-with-wgan/blob/master/image/1_0.jpg?raw=true"></img>

#### 第52000次迭代
<img src="https://github.com/zjq1996518/srgan-with-wgan/blob/master/image/1_2000.jpg?raw=true"></img>

#### 第54000次迭代
<img src="https://github.com/zjq1996518/srgan-with-wgan/blob/master/image/1_4000.jpg?raw=true"></img>

可以看到随着训练的增加，生成的图片质量越来越清晰
