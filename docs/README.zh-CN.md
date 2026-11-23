# 中文使用指南

本项目提供 SO101 的“采集演示 → 数据转换 → Flow Matching 模仿学习训练 → 离线检查 → 实机部署”流程。
模型来自 ReinFlow 的相关实现，保留来源和许可证；这里实现的是模仿学习阶段。

## 支持范围

- 已实际开发使用：Hiwonder SO101、HX-30HM、主从臂、单腕部 RGB 相机。
- STS3215 可通过配置选择，但本项目未完成该硬件的实机验证。
- Linux CUDA 用于训练；macOS CPU 用于本机推理。预览使用 Tk/Pillow。
- 相机停帧、夹爪过载等问题仍需结合 USB、校准和机械负载排查。

## 安装与配置

训练环境：

```bash
python -m pip install -e '.[prepare,test]'
```

硬件环境额外安装 `requirements-hardware.txt` 指定的 Hiwonder LeRobot 分支。
建议与服务器训练环境分开。复制 `configs/hardware.example.yaml` 为
`hardware.local.yaml`，填写真实串口、校准 ID 和相机编号；本地配置被 Git 忽略。

## 采集

```bash
python -m scripts.hardware --config hardware.local.yaml calibrate-leader
python -m scripts.hardware --config hardware.local.yaml calibrate-follower
python -m scripts.hardware --config hardware.local.yaml teleoperate
python -m scripts.hardware --config hardware.local.yaml record \
  --name pick_place --episodes 30 --task 'Pick up the object and place it in the container.'
```

每条最长 30 秒、条间复位 15 秒。方向右键提前结束当前条，左键重录，Esc 停止；
macOS 的终端需要相应键盘监听权限。已有目录不会被覆盖，续录条数表示“新增条数”：

```bash
python -m scripts.hardware --config hardware.local.yaml record \
  --resume datasets/pick_place --episodes 10 \
  --task 'Pick up the object and place it in the container.'
```

录制和部署不要同时占用同一个相机/串口。发生图像超时不会用旧图像继续录制。

## 转换和训练

```bash
python -m runner.flow_matching.prepare \
  --dataset datasets/pick_place --out data/pick_place --val-episodes 3
python -m runner.flow_matching.train \
  --data data/pick_place --device cuda:0 --epochs 200 --batch-size 64
```

30 条演示中 27 条训练、3 条验证，按整条轨迹划分；只用训练集计算归一化统计。
预处理将 RGB 双线性缩放为 96×96，保存 CHW uint8。模型输入图像和 6 维关节状态，
预测未来 16 帧动作。前五维为角度，夹爪为校准后的 0～100 标度。

`result/flow_matching/` 保存模型、完整配置、统计参数、数据说明和训练日志。
验证损失不能直接视为抓取成功率。换数据集后必须使用新训练产生的整套部署文件。

## 部署

```bash
# 先用验证图像离线检查，不连接硬件。
python -m runner.flow_matching.predict --run result/flow_matching/YOUR_RUN \
  --data data/pick_place/val.npz

# 默认只读，打开预览，不发送动作或配置扭矩。
python -m deployment.deploy --run result/flow_matching/YOUR_RUN \
  --port YOUR_PORT --robot-id follower

# 检查通过、工作区准备好后，显式启动 5 秒执行。
python -m deployment.deploy --run result/flow_matching/YOUR_RUN \
  --port YOUR_PORT --robot-id follower --execute --duration 5
```

公开版本与个人本机脚本的区别：**默认只读**，执行必须加 `--execute`。
Ctrl+C、Esc 或关闭预览请求停止，退出尝试关闭扭矩。先托稳机械臂，避免释放后下落。
程序退出不等于已确认所有舵机释放；有故障时单独检查或释放：

```bash
python -m deployment.deploy --port YOUR_PORT --robot-id follower --diagnose
python -m deployment.deploy --port YOUR_PORT --robot-id follower --release
```

如果过载持续，托稳后切断舵机供电、卸掉负载并检查夹爪。不要强扳关节或屏蔽故障位继续执行。
软件的限速、限位和故障检查不能代替硬件急停。

完整说明见仓库根目录 [README](../README.md) 和 [硬件排错](hardware.md)。
