## **Learning the Degradation Distribution for Blind Image Super-Resolution论文程序**

The codes are built on the basis of [BasicSR](https://github.com/xinntao/BasicSR).

**1、KernelModel:**

![image-20220409114809532](https://gitee.com/xjg0216/blogimg/raw/master/img/image-20220409114809532.png)

核函数的大小为（21， 21）

输入通道： nc_in = nc = 3

nf：隐藏层通道数

nb： ResBlock个数 为 8

```python
        deg_kernel = [
            nn.Conv2d(3, 64, 1, 1, 0),
            nn.BatchNorm2d(64), nn.ReLU(True),
            *[
                ResBlock(nf=64, ksize=1)
                for _ in range(8)
                ],
            nn.Conv2d(64, 21 ** 2, 1, 1, 0),
            nn.Softmax(1)
        ]
        self.deg_kernel = nn.Sequential(*deg_kernel)
```

8个ResBlock

![image-20220409115439430](https://gitee.com/xjg0216/blogimg/raw/master/img/image-20220409115439430.png)

****

**2、NoiseModel**

一样的套路

![image-20220409122542725](https://gitee.com/xjg0216/blogimg/raw/master/img/image-20220409122542725.png)

```python
deg_noise = [
            nn.Conv2d(in_nc, nf, head_k, 1, head_k//2),
            nn.BatchNorm2d(nf), nn.ReLU(True),
            *[
                ResBlock(nf=nf, ksize=body_k)
                for _ in range(nb)
                ],
            nn.Conv2d(nf, opt["dim"], 1, 1, 0),
        ]
        self.deg_noise = nn.Sequential(*deg_noise)
```

输入通道： in_nc = nc = 3

nf：隐藏层通道数32

nb： ResBlock个数 为 8

**3、PatchGANDiscriminator**(netD)

```
PatchGANDiscriminator(
  (model): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (3): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (4): LeakyReLU(negative_slope=0.2, inplace=True)
    (5): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (7): LeakyReLU(negative_slope=0.2, inplace=True)
    (8): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (10): LeakyReLU(negative_slope=0.2, inplace=True)
    (11): Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)
```

<font color=red>前向传播</font>

```python
    def deg_forward(self):
        (
            self.fake_real_lr,
            self.predicted_kernel,
            self.predicted_noise,
         ) = self.netDeg(self.syn_hr)
```

**self.netDeg：**

* DegModel网络实例化

**self.syn_hr**：

* 高分辨率图像（HR），采用的batch为32， 所以self.syn_hr的shape为（32， 3， 128， 128）

**self.fake_real_lr：**

* 生成的低分辨率图像（LR）， 大小为（32， 3， 32， 32）

**self.predicted_kernel**

* 模糊核，大小为（32， 21， 21）

**self.predicted_noise**

* 噪声， 大小为（32， 1， 1， 1）

**********

<font color=red>进入DegModel网络中</font>

```
        B, C, H, W = inp.shape
        h = H // self.scale
        w = W // self.scale
```

H， W降4倍----->（32， 32）； self.scale = 4

初始化$z_k$:

```python
zk = torch.randn(B, self.opt["nc"], 1, 1).to(x.device)
```

大小为(32, ,3, 1, 1)

```python
kernel = self.deg_kernel(inp).view(B, 1, ksize**2, *inp.shape[2:])
```

inp就是上面的zk， 将它作为Kernel的输入

**kernel输出大小为(32, 1, 441, 1, 1)，得到的kernel与输入相卷积，具体如下：**

```
x = x.view(B*C, 1, H, W)   ##x为HR图像， 先转为(32x3, 1, 128, 128)
x = F.unfold(
     self.pad(x), kernel_size=ksize, stride=self.scale, padding=0
     ).view(B, C, ksize**2, h, w)  # 手动实现滑动窗口的操作，只有卷，没有积 输出大小为(32, 3, 441, 32, 32)

x = torch.mul(x, kernel).sum(2).view(B, C, h, w) ##(32, 3, 32, 32)
kernel = kernel.view(B, ksize, ksize, *inp.shape[2:]).squeeze() # (32, 21, 21)
return x, kernel
```

**********

下面再添加噪声

```python
noise = self.deg_noise(x.detach())
x = x + noise
```

```
zn = torch.randn(x.shape[0], self.opt["nc"], 1, 1).to(x.device) #(32, 3, 1, 1)
```

noise的输出大小为(32, 1, 1, 1), 再将噪声加入到x中

```python
return x, kernel, noise
x: (32, 3, 32, 32)
kernel:(32, 21， 21)
noise(32, 1, 1, 1)
```

计算生成的LR图像与测试中真实的LR图像的对抗损失

```python
g1_adv_loss = self.calculate_gan_loss_G(self.netD1, self.losses["lr_adv"], real, fake)
def calculate_gan_loss_G(self, netD, criterion, real, fake):

        d_pred_fake = netD(fake)#fake:(32, 3, 32, 32);    d_pred_fake: (32, 64, 8, 8)
        loss_real = criterion(d_pred_fake, True, is_disc=False)  #d_pred_fake与相同大小的target_label计算损失，target_label为全1；；；；MSELoss()

        return loss_real
```

再加上正则项损失

```python
loss_G += self.loss_weights["noise_mean"] * noise_mean
```

上面是固定的判别器，然后再训练判别器

前向传播到SR，SR用的<font color=red>EDSR</font>

```
if self.optim_sr:
      loss_dict = self.optimize_sr_models(step, loss_dict)
```

输入低分辨率图像（32， 3， 32， 32），推理高分辨率图像（32， 3， 128， 128）

```
self.syn_sr = self.netSR(self.fake_real_lr_quant.detach())
```
