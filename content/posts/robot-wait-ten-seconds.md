+++
title = '机器人怎么知道自己已经等了十秒？'
date = 2026-11-25T14:57:00+08:00
draft = false
tags = ['embodied-intelligence', 'data-quality', 'vla']
description = 'OpenWAM 会少采样首步静止的窗口。碰到必须保持十秒的工序，时间该交给数据、模型，还是执行层？'
+++

OpenWAM 的 RoboTwin 微调配置里有个开关，叫 `filter_static_segments`。我看到这一行的时候想，有的停顿本来就是任务的一部分。比如压住一个零件，保压十秒，再抬手。把这段数据清掉，模型还有机会学到这道工序吗？把十秒全留着，它又怎么知道自己已经等够了？

## OpenWAM 怎么处理静止

这里的过滤发生在训练采样时。dataloader 拿窗口里的第一步动作和当前 `proprio` 比较。归一化后，各维的绝对差都小于 `1e-5` 就重抽，最多三次。抽不到合适的，原窗口照用。判断只看第一步，所以即使一个窗口后面会抬手，只要开头在保持，也可能被换掉。 [默认配置](https://github.com/OpenWAM-Official/OpenWAM/blob/66b047b9e2ef03c0fe5e9b30fe63c04c31b5a574/configs/dataloader/robotwin.yaml#L52-L59) · [采样代码](https://github.com/OpenWAM-Official/OpenWAM/blob/66b047b9e2ef03c0fe5e9b30fe63c04c31b5a574/openwam/dataloader/robotwin.py#L838-L898)

[![OpenWAM 静止窗口判定示意。第一步就动的窗口照常使用。第一步保持的两种窗口都会尝试换样，即使其中一条后面会抬手。](/images/robot-wait-ten-seconds/static-filter.png)](/images/robot-wait-ten-seconds/static-filter.png "查看大图")

OpenVLA 的作者遇到过保留 no-op 的后果。他们用原始 BridgeData V2 训练单步策略，模型在评估时经常输出全零动作，机械臂就一直待在原地。示教员停下来换一下手势，留这样的数据确实可能有麻烦。保压也会录出一段相似的数据，只是拿掉这十秒，工序就变了。 [OpenVLA 论文附录 C](https://arxiv.org/html/2406.09246v3#A3)

## 十秒留在数据里，模型看得见吗？

先算一下窗口。OpenWAM 的 RoboTwin 默认预测 32 步。拿 20 Hz 举例，32 步覆盖 1.6 秒。第 0 步压到位，第 200 步第一次抬手。从每帧往后取 32 步动作，前 168 个窗口全是保持。到 8.4 秒，抬手才第一次出现在窗口末尾。 [窗口配置](https://github.com/OpenWAM-Official/OpenWAM/blob/66b047b9e2ef03c0fe5e9b30fe63c04c31b5a574/configs/dataloader/robotwin.yaml#L24-L30)

[![20 Hz 下的十秒保压。从第 0 步到第 167 步开始的窗口未来 32 步全是保持，第 168 到 199 步开始的窗口能看到第 200 步的抬手。](/images/robot-wait-ten-seconds/sliding-windows.png)](/images/robot-wait-ten-seconds/sliding-windows.png "查看大图")

麻烦在于，第 1 秒和第 9 秒的图像、位姿、语言指令可能都差不多。前者该继续压，后者的 chunk 里该有抬手。每次重规划都只看当前输入的话，模型拿不到「已经压了几秒」。如果它这一轮选了保持，下一轮又看到同样的状态，就有可能反复选保持。

如果策略能在刚压到位时就生成整个等待段，且一直执行到抬手，一个 chunk 也可以把这件事做完。ACT 原论文谈过这种情况。实际要算的是执行了多少步才重新规划，因为执行器可以只取预测序列的前一段。 [ACT 论文 §IV-A](https://arxiv.org/pdf/2304.13705#page=4) · [OpenWAM 同步执行器](https://github.com/OpenWAM-Official/OpenWAM/blob/66b047b9e2ef03c0fe5e9b30fe63c04c31b5a574/openwam/deploy/executors/sync_executor.py#L44-L67)

短 chunk 的策略卡住，有过类似的实验。Chen 等人在 ALOHA 仿真里用未过滤的示教训练策略，每次预测 50 步、执行 20 步。**84.9% 的失败回合**里，机械臂出现了停滞，反复做相似的小动作。这是抓取、插入附近的实验。套到这里，十秒保压会不会让 OpenWAM 卡住，仍然是推测。 [策略停滞实验，表 I–II](https://arxiv.org/html/2508.15669#S5)

## 三种做法

[![三种实现路径。从上到下，任务执行层到点换指令，设备把完成状态变成相机可见的信号，以及通过改输入、记忆或输出来改模型。](/images/robot-wait-ten-seconds/control-interfaces.png)](/images/robot-wait-ten-seconds/control-interfaces.png "查看大图")

**一，到点换指令。** 同一条示教按阶段标两句话，「压到位，保持」和「抬手」。训练时把抬手动作归到第二句。部署时程序确认压到位，开始计时，到点再下发「抬手」。切换时把缓冲区里还没执行的保持动作清掉，否则新指令发了，机械臂可能还在执行旧 chunk。要同时看计时和设备状态，也可以让 System 2 读这些信号，负责发哪句指令。π₀.₅ 的高层输出语言子任务、低层生成动作，提供了一个类似的接口。 [π₀.₅ 论文](https://arxiv.org/html/2504.16054)

**二，让设备亮个灯。** 红灯继续保压，绿灯抬手。绿灯一直亮到这一轮结束，相机错过变灯的那一帧也没关系。训练时可以把红灯窗口的动作监督截到变灯处，让抬手从看到绿灯的窗口开始学。测试的时候故意把灯调早或调晚一点，就能看出模型是不是在用这个信号。

**三，改模型。** 最简单是给 System 1 加一个输入，告诉它从压到位到现在过了几秒。表还是部署进程在走，模型拿这个数决定接着压还是抬手。也可以让模型跨 chunk 记住「压到位」这个事件。要用记忆判断真实的秒数，得把时间戳或每次更新的间隔带进去。SimpleMemVLA 的做法可以拿来参考，它把带时间戳的历史画面送进模型。 [SimpleMemVLA 论文](https://arxiv.org/abs/2609.05533) · [开源实现](https://github.com/OpenBMB/SimpleMemVLA)

还有一种是改输出，让 VLA 能发出 `HOLD(10s)`。普通的 action chunk 是一串连续动作，要发这样的命令，就得改输出格式、做对应的训练标签，再让执行器认得它。计时和保压仍然归执行器。到点后执行事先排好的抬手动作，或者给模型发一个完成事件。想往下看，可以读 Sutton 等人关于 *options* 的工作。 [时间抽象与 options](https://ics.uci.edu/~dechter/courses/ics-295/winter-2018/papers/Sutton-Precup-Singh-AIJ99.pdf)

## 怎么测它到底在等什么

我想先测一个小变化。第一次第 2 秒压到位，第二次第 5 秒才压到位，两次都要求保压十秒。要是两次都在全程第 12 秒抬手，第二次实际只压了七秒。再把要求换成保压 3 秒和 10 秒，记录每次从压力到位到实际抬手的间隔，以及提前抬手和超时的次数。

[![两条任务时间线。第一回合第 2 秒压到位、第 12 秒抬手，第二回合第 5 秒压到位、第 15 秒抬手。两次都实际保持 10 秒。](/images/robot-wait-ten-seconds/timing-test.png)](/images/robot-wait-ten-seconds/timing-test.png "查看大图")

*Is Your Robot’s Memory Just a Clock?* 做过一个适合参考的对照。重复抓取时，作者只改历史帧的时间戳，画面不动，策略就会提前结束或多抓几次。在真正要求计时的 `TimedTransfer-Medium` 里，原设置 50 次全成功，把时间戳拨快，50 次都过早，拨慢则都过晚。重复抓取要记的是抓了几次，保压要记的是从哪一刻开始压、已经压了多久。 [论文 §5.2、附录 J](https://arxiv.org/html/2611.14488)

回到数据清洗。如果采集时能记下什么时候到位、开始保压、允许释放，后面就好办多了。拿到一段不动的轨迹，可以回去看当时在等什么，再决定这段要不要留、该怎么切窗口。
