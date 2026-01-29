# 快速开始 (Quickstart)
##  运行 (Run)

### 启动主程序
```bash
./build/standard configs/standard3.yaml
```

### 单元测试

#### 相机测试
验证相机是否能正常出图：
```bash
./build/camera_test --config-path=configs/standard3.yaml --display
```

#### 通信测试
验证与C板通讯（打印欧拉角+弹速）：
```bash
./build/cboard_test configs/standard3.yaml
```

## 设置开机自启 (Autostart Setup)

本项目包含 `watchdog.sh` (守护进程) 

### 步骤 1: 检查脚本权限
确保脚本具有执行权限（如果之前未设置）：
```bash
chmod +x /home/setsuna/RM/AutoAim/sp_vision_25/watchdog.sh
```

### 步骤 3: 验证
重启或注销登录后，打开终端使用 screen 查看运行状态：
```bash
# 进入会话查看实时日志
journalctl -u sp_vision.service -f
```

### 步骤 4: 停止自启程序
如果需要停止后台运行的自瞄程序：
```bash
pkill -f watchdog.sh
pkill -f standard
# 或者在 screen 会话中按 Ctrl+C
```
### 在 watchdog.sh 中添加 ROS 2 环境变量
在 `watchdog.sh` 脚本的开头部分添加以下内容，以确保
# Source ROS 2 environment (Jazzy)  记得修改为你的 ROS 2 版本路径

if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
fi

### 修改开机自启动后，请重新执行以下命令以应用更改：
```bash
sudo systemctl daemon-reload
sudo systemctl restart sp_vision.service
# 验证服务状态
sudo systemctl status sp_vision.service
```

### 修改源代码后记得重新编译项目：
```bash
mkdir -p build
cd build
cmake ..
make standard_mpc_se -j4
### 这里以 standard_mpc_se 为例，根据需要编译相应的可执行文件
```
```bash
## 或者直接
cmake --build build 
```