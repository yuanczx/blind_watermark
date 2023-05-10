# 算法复现：基于DCT的数字水印隐藏方法

## 文件结构
- `img` 文件夹：存放程序使用到的图片。
  - `GT` 文件夹：我们自己拍摄的图像。
  - `MIT-Adobe_FiveK` 文件夹：取自[MIT-Adobe FiveK dataset](https://data.csail.mit.edu/graphics/fivek/)中的图片
- `out`：文件夹：存放程序运行产生的图片。
- `watermark.py`：水印嵌入与提取程序。
- `test_1.py`：测试脚本1。
- `test_2.py`：测试脚本2。
- `test_3.py`：测试脚本3。
- `attack.py`：一些常见的噪声攻击。

## 结果复现
- 安装使用到的库 `pip install -r requirements.txt`
- 执行 `python test_1.py` 查看不用图片嵌入水印后提取出水印的对比。
- 执行 `python test_2.py` 查看对嵌入水印后的图像进行滤波对提取水印的影响。
- 执行 `python test_3.py` 查看对嵌入水印后图像加入噪声对提取水印的影响