# Problem 030 - bytetriper 的数字识别（寒假 ver）

**ACMOJ Problem ID**: 2569

## Table of Contents

- [Problem 030 - bytetriper 的数字识别（寒假 ver）](#problem-030-bytetriper-的数字识别寒假-ver)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Background](#background)
  - [Assignment Description](#assignment-description)
    - [Grade Composition](#grade-composition)
  - [Assignment Requirements](#assignment-requirements)
    - [Input Format](#input-format)
    - [Output Format](#output-format)
    - [Samples](#samples)
    - [Data Constraints](#data-constraints)
  - [Per-Testcase Resource Limits](#per-testcase-resource-limits)
  - [Test Data](#test-data)
  - [Submission Requirements](#submission-requirements)
    - [Submission Format](#submission-format)
    - [Submission Guidelines](#submission-guidelines)
    - [Evaluation Notes](#evaluation-notes)
    - [Academic Integrity](#academic-integrity)

## Introduction

### Background

Implement handwritten digit recognition without neural networks. Open-ended CV problem.

## Assignment Description

### Grade Composition

| Grading Component | Percentage |
| :--: | :--: |
| Pass **2569. bytetriper 的数字识别（寒假 ver）** | 80% |
| Code Review | 20% |

Here are several points that need clarification:

- In the Code Review, we will **strictly examine your code style and repository organization structure, etc.**

- This assignment provides some sample data for testing, stored in the `/workspace/data/030/data_test/` directory. Note that these are not the test cases on the Online Judge. Passing all local test cases does not guarantee that you will pass the OJ tests.

- Besides the provided sample data, we also encourage you to design your own test data based on your program logic to assist debugging.

## Assignment Requirements

### Problem Description

*本题来源于之前的 PPCA，你不应该期待满分。本题寒假版满分为 200 分。*

RT.

图像识别一直是很热门的话题，因此跟热点的 bytetriper 这次任务是做简单的手写数字识别！

大家都知道，AI 已经杀死了 CV（计算机视觉），但守旧的 bytetriper 不喜欢神经网络，因此决定把这个任务出成开放题，交给各位解决。
### 禁止在judge中进行IO/文件IO
### 任务描述

在本题中，你只需要解决基础的灰度图片单个数字识别的问题。即给定一张灰度图片，输出 0-9 的一个数字作为你的预测。

例如：
!["6"](https://raw.githubusercontent.com/bytetriper/img/main/6.png) 应该被分类为 6，

![难以辨认的 4](https://raw.githubusercontent.com/bytetriper/img/main/8.png) 应该被分类为 4（虽然很像 9）。

具体来讲，你需要实现 `src.hpp` 中的 `judge` 函数并将整个 `src.hpp` 上交到 OJ。
   
```cpp
typedef std::vector<std::vector<double>> IMAGE_T;
int judge(IMAGE_T &img);
```

其中 `IMAGE_T` 为一个二维 vector，即图片的二维数组表示，数据存储约定见后。

`judge` **函数的返回值即为你对于这张图片的预测值。**

**请注意，返回除了 0-9 之外的任何其他返回值都被认为是非法的，会引发错误。**

### 数据

下发数据和代码都已放在 [Code Repo](https://github.com/bytetriper/Number-Recognition)。

我们对图片的存储方式做如下规定：

* 所有图像均被表示为二维数组 $A_{N\times N},A_{i,j}\in [0,1]$ ，其中 1 为纯白，0 为纯黑。
* 所有图像都由 png/ppm 格式储存。
* 所有图像（包括测试集）都是$28\times 28$的图片。
* 数字都为白色，且符合人类正常书写规律。不会出现倒着写，横着写的情况。
* 保证数字一定存在于图片中，但不对数字的形状、大小、区域做任何保证。
* 部分数据已经下发，详见仓库中的 `data/`，数据具体来源及与测试数据集差别详见 tips：


### Prerequisite

* 由于使用 C++ 读入 png/jpg/jpeg 图片非常不方便，我们下发了 ppm 格式的图片以方便读取。
* 同时，我们也给出了读取 PPM 文件的代码，你可以直接使用。
* 虽然文件以 PPM 格式存储，但内容实际是 P5 的 PGM，即灰度图。你可以在[这里](http://netpbm.sourceforge.net/doc/pgm.html)查看 PGM 的具体格式。
  
#### Read from ppm

* 源文件 `read_support.cpp` 与 `read_support.h` 已下发在仓库中。
`read_support.h` 中提供了两个函数的接口：

```cpp
typedef std::vector<std::vector<double>> IMAGE_T;
IMAGE_T read_ppm(const std::string &filename);
IMAGE_T read_ppm();
void check(IMAGE_T mat);
```

* `read_ppm` 函数接受一个字符串作为 ppm 文件地址，返回一个二维 vector，即图片的二维数组表示，数据存储约定见前。没有参数的 `read_ppm` 函数行为和上述函数相同，但是会从标准输入读取 ppm 文件。
* `check` 函数接受一个二维 vector，并在控制台输出该二维 vector 的内容，以方便你调试。


### 评分

本题评分标准如下：

我们希望衡量的是算法的正确的概率。设测试数据集有 $N$ 张图片，你正确判断了其中的 $M$ 张，那么你的得分是

$$
200 \times \frac{M}{N}.
$$

上交到 OJ 的代码会在测试集数据下进行测试。请注意：**测试数据集与下发数据不同**。

**为了方便更有针对性的优化**，在评测信息中给出了 $10\times 10$ 的矩阵 $Acc$ ，其中 $Acc_{i,j}$ (0-based) 为你在 label $i$ 的测试数据上的预测为 label $j$ 的概率（ $\sum_{j=0}^9 Acc_{i,j}=1$ ）。

### 本地测试

欢迎在本地用下发数据自行评测结果，具体评测方式：

* 进入仓库根目录（和 `Makefile` 同级），执行 `make`，编译出可执行文件 `test`。由于 stdc++17 在 MINGW 上的 path 实现有一些问题，请尽量使用 Linux 或者 WSL 进行测试。
  > 如果你没有安装 make，可以使用命令 `g++ -std=c++17 test.cpp read_support.cpp -o test` 编译。
* 如果希望简单地在所有下发数据上测试，请使用：`./test`，输出为在所有数据上的正确率。
* 如果希望对某个 label 的数据进行测试，请使用：`./test [label]`（如 `./test 1` ），输出为在该 label 上的正确率。
* 你也可以自行改动代码以实现更精细的评测。

### Reminders

鉴于 bytetriper 希望这是一道人类智慧题，而不是人工智能题，有一些 tips 送给大家：

* 测试数据集和下发的数据都随机采样自同一个数据集 MNIST，这意味他们在特征上基本相同，同时，测试集的大小是下发数据集的 60 倍。由于测试集和下发集不相同，你的算法在本地上跑出的结果很可能不同于 OJ 上的结果。
* **禁止上网查询信息**，同时，禁止直接使用任何图像处理库的代码，包括但不限于 OpenCV，PIL，skimage 等。由于 OJ 本身没有相关库支持，因此这里的禁止使用指的是不能抄袭相关库实现。
* 对于依旧使用人工智能方法的同学，我们会对准确率高的代码手动检查，请特别注意 PPCA 网络限制包括但不限于：
   1. **不允许使用任何外部数据**
   2. **不允许使用任何预训练模型**
* 由于以图片格式查看二进制 PPM 文件会碰到一些困难，（使用 VSCode 的）同学可以下载 VSCode 的插件来查看 PPM 文件，这是被允许的。同时，有些浏览器原生支持 PPM 格式，也可以使用浏览器打开 PPM 文件。
* 如果实在无法查看 PPM，可以查看 `png_data/` 下的 png 格式文件，这与 ppm 是完全相同的。
* 对于没有接触过机器学习的同学的提示：
	1. 一百张图片的数据量是非常小的，请尽量不要all in 自己没接触过的机器学习算法
	2. 可以通过观察下发数据多考虑数字本身的特征以及`分布`
	3. **bytetriper希望这是一道人类智慧题，而不是人工智能题**

### Input Format

See the problem description above.

### Output Format

See the problem description above.

### Samples

No sample data provided for this problem.

### Data Constraints

See the problem description for constraints.

## Per-Testcase Resource Limits

- **Time Limit (per test case)**: 30000 ms
- **Memory Limit (per test case)**: 1024 MiB
- **Disk Usage**: No disk usage is permitted.

## Test Data

The test data for this problem is located at `/workspace/data/030/data_test/`.

## Submission Requirements

### Submission Format

Submit a single C++ source file to the Online Judge. The OJ provides its own `main.cpp` which `#include "src.hpp"`. Your submission will be written to `src.hpp` and compiled together with the OJ's driver code.

### Submission Guidelines

- The submitted code must be able to compile successfully through the above compilation process
- The program needs to be able to read data from standard input and write results to standard output
- Please ensure the code runs correctly within the given time and space limits
- **You must use C++ or C language** to implement this assignment

### Evaluation Notes

- The evaluation system will test your program using the provided test data
- The program output must exactly match the expected output (including format)
- Exceeding time or memory limits will be judged as the corresponding error type
- Please pay attention to the overall time performance of your code and the time complexity of each part of your algorithm.

### Academic Integrity

If any violations are found during evaluation or code review (including but not limited to using unconventional methods to pass test cases), your final score may be significantly reduced or become **0 points**.
