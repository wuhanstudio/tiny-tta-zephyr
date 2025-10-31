# TinyTTA (Zephyr)

> TinyTTA Engine is a lightweight framework for enabling Test-Time Adaptation (TTA) on edge devices like microcontrollers (MCUs).

The original implementation uses ARM MbedOS, which will reach end of life in July 2026.

- TTE: https://github.com/h-jia/TTE

## Quick Start

Step 1: Install Zephyr

```
$ pip install west
$ west init ~/zephyrproject
$ cd ~/zephyrproject
$ west update
$ west zephyr-export
$ pip install -r ~/zephyrproject/zephyr/scripts/requirements.txt
$ echo "source ~/zephyrproject/zephyr/zephyr-env.sh" >> ~/.bashrc
```

Step 2: Install Zephyr-SDK

```
$ cd ~/zephyrproject/zephyr
$ west sdk install
```

Step 3: Compile

```
$ git clone https://github.com/wuhanstudio/tiny-tta-zephyr/
$ cd tiny-tta-zephyr

$ west build -b stm32f103_mini -p
$ west flash
```
