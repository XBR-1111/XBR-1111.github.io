+++
title = '机器人怎么知道自己已经等了十秒？'
date = 2026-11-25T14:57:00+08:00
draft = false
tags = ['embodied-intelligence', 'data-quality', 'vla']
description = '从 OpenWAM 的静止过滤规则出发，聊聊一台只会输出 action chunk 的机械臂，怎么做完一道需要保压十秒的工序。'
+++

假设你要教机械臂做一道工序。压住一个零件，保持十秒，然后抬手。示教的时候，你把手柄稳住，心里数到十，再往上一提。这条轨迹就录完了。

等数据拉出来一看，中间那十秒的画面几乎没变化，关节位置也差不多。要是数据管线里正好有一条规则叫「过滤静止帧」，这十秒该怎么办？

## 被过滤的是哪一段

我是在翻 OpenWAM 的 RoboTwin 微调代码时想到这件事的。默认配置里，dataloader 比较的是窗口里**第一步动作**和当前的机器人状态。归一化后，如果每一维的绝对差都小于 `1e-5`，就随机找一个窗口来替换。最多找三次，找不到合适的就用回原来的窗口。这个判断只用在训练采样时。 [配置](https://github.com/OpenWAM-Official/OpenWAM/blob/66b047b9e2ef03c0fe5e9b30fe63c04c31b5a574/configs/dataloader/robotwin.yaml#L52-L59) · [代码](https://github.com/OpenWAM-Official/OpenWAM/blob/66b047b9e2ef03c0fe5e9b30fe63c04c31b5a574/openwam/dataloader/robotwin.py#L838-L898)

[![OpenWAM 静止窗口判定示意。第一步就动的窗口照常使用。第一步保持的两种窗口都会尝试换样，即使其中一条后面会抬手。](/images/robot-wait-ten-seconds/static-filter.png)](/images/robot-wait-ten-seconds/static-filter.png "查看大图")

这条规则有来由。OpenVLA 的作者在原始 BridgeData V2 上遇到过一个问题。数据里有不少全零动作，模型会在评估里反复输出全零，机械臂就卡在原地。示教员停下来换个手势，确实可能录下一堆这种样本。可要是机械臂正在保压，同样的「没动」就有工艺意义。看第一步动作，很难把这两件事分开。 [OpenVLA 论文附录 C](https://arxiv.org/html/2406.09246v3#A3)

## 那就把十秒全留下来？

OpenWAM 的 RoboTwin 默认窗口预测 32 步。用 **20 Hz** 举例，一个 chunk 对应 1.6 秒。这里把压到位的观测记作第 0 步，第一次抬手的动作记作第 200 步。从当前观测往后取 32 步动作，前 168 个起点看到的标签都是「继续保持」，最后 32 个起点的标签里才看得到抬手。 [窗口配置](https://github.com/OpenWAM-Official/OpenWAM/blob/66b047b9e2ef03c0fe5e9b30fe63c04c31b5a574/configs/dataloader/robotwin.yaml#L24-L30)

[![20 Hz 下的十秒保压。从第 0 步到第 167 步开始的窗口未来 32 步全是保持，第 168 到 199 步开始的窗口能看到第 200 步的抬手。](/images/robot-wait-ten-seconds/sliding-windows.png)](/images/robot-wait-ten-seconds/sliding-windows.png "查看大图")

现在挑两个时刻看。刚压到位没多久，应该再保持一整段。快到十秒了，应该再保持几步，然后抬手。对一个每次只拿当前画面、位姿和同一句指令的策略来说，这两次收到的输入可能几乎一样。

如果它每次都选了保持，执行完一个 chunk，回来看到的还是原来的样子，就可能接着保持。另一方面，要是策略能在压到位时一次生成覆盖完整等待的 chunk，并且一路执行到抬手，这件事在单个 chunk 里就能做完。ACT 原论文提过，动作分块能缓解落在同一个 chunk 内的停顿。实际部署还要看每次执行多少步，OpenWAM 的同步执行器可以只执行生成序列的前一段。 [ACT 论文 §IV-A](https://arxiv.org/pdf/2304.13705#page=4) · [OpenWAM 同步执行器](https://github.com/OpenWAM-Official/OpenWAM/blob/66b047b9e2ef03c0fe5e9b30fe63c04c31b5a574/openwam/deploy/executors/sync_executor.py#L44-L67)

带 chunk 的策略确实也会卡住。Chen 等人在 ALOHA 仿真里用未过滤的示教训练策略，预测 50 步，每次执行 20 步。84.9% 的**失败回合**出现了停滞，机械臂在某些位置反复做相似的小动作。他们观察的是抓取、插入附近的停滞。OpenWAM 跑本文这道十秒保压工序时会不会卡住，我还没做过实验。 [策略停滞实验，表 I–II](https://arxiv.org/html/2508.15669#S5)

## 怎么把十秒接进系统

[![三种实现路径。从上到下，任务执行层到点换指令，设备把完成状态变成相机可见的信号，以及通过改输入、记忆或输出来改模型。](/images/robot-wait-ten-seconds/control-interfaces.png)](/images/robot-wait-ten-seconds/control-interfaces.png "查看大图")

**一，到点换一句指令。** 训练时把示教按阶段标成「压到位并保持」和「抬手，继续下一步」，让抬手动作归到第二条指令。上线后，业务代码在确认压到位时开始计时。到点后切指令，清掉还没执行的保持动作，让 VLA 重新出一个 chunk。如果还要等设备完成等信号，可以由 System 2 读工序状态和计时器，再发下一条指令。这样的分层接口也可以参考 π₀.₅，论文里的高层先给出语言子任务，低层再生成动作。 [π₀.₅ 论文](https://arxiv.org/html/2504.16054)

**二，让设备给一个看得见的信号。** 红灯时保持，达到释放条件就亮绿灯。绿灯保持到这一轮结束，错过变灯的那一帧也能看懂。训练时可以在变灯的地方截断红灯窗口的动作监督，让抬手从看到绿灯的窗口开始学。工程上，我会让测试里的灯有时早亮、有时晚亮，看机械臂到底在看灯，还是在复现示教的节奏。

**三，改模型。** 最直接的是加一个输入。执行层确认压到位后开表，把已经等了几秒传给 VLA，它再决定接下来是保持还是抬手。也可以加跨 chunk 的时序记忆，让模型把「压到位」这件事带到下一次推理。要按实际秒数做决策，还得提供时间戳或每次更新的间隔。SimpleMemVLA 有个可以翻代码的做法，把带时间戳的历史画面送进模型。 [SimpleMemVLA 论文](https://arxiv.org/abs/2609.05533) · [开源实现](https://github.com/OpenBMB/SimpleMemVLA)

改模型也可以从输出入手，例如把 `HOLD(10s)` 加进动作表示。这就得加对应的训练标签，并且实现一个能解析命令的执行器。收到命令后，执行器负责保压和计时，到点后执行预排的抬手动作，或者把完成事件传给模型。这类带终止条件、能执行一段时间的动作，可以顺着 Sutton 等人关于 *options* 的论文往下看。 [时间抽象与 options](https://ics.uci.edu/~dechter/courses/ics-295/winter-2018/papers/Sutton-Precup-Singh-AIJ99.pdf)

## 假如它只记住了第十二秒

如果往下做实验，我会让机械臂有时第 2 秒才压到位，有时第 5 秒才压到位。两次都保压十秒，抬手就应该分别出现在第 12 秒和第 15 秒。再让同一套策略按要求压 3 秒或 10 秒。每次记录从压力达到目标到实际抬手的间隔，看看它会提前多少、拖多久，有多少次一直没抬。

[![两条任务时间线。第一回合第 2 秒压到位、第 12 秒抬手，第二回合第 5 秒压到位、第 15 秒抬手。两次都实际保持 10 秒。](/images/robot-wait-ten-seconds/timing-test.png)](/images/robot-wait-ten-seconds/timing-test.png "查看大图")

有篇论文已经把这里的区别做成了实验。*Is Your Robot’s Memory Just a Clock?* 在重复抓取的任务里只改历史帧的时间戳，画面原样保留，策略就会提前结束或多做几次。但在真正按时刻操作的 `TimedTransfer-Medium` 里，时间本来就是题目的一部分。原设置 50 次全成功，时间戳加速后 50 次都过早，减速后 50 次都过晚。同一块表，在重复抓取里可能成了捷径，在保压工序里却是你特地要给机器人的东西。 [论文 §5.2、附录 J](https://arxiv.org/html/2611.14488)

数据侧也可以接着做。采集时记下「到位」「开始保压」「允许释放」这几个事件和时间戳。等数据进了清洗管线，至少还能问回去，当时为什么停在这儿？

下次看到一段几乎一模一样的画面，我会先问一句。**如果把这十秒剪掉，这道工序还算做完了吗？**
