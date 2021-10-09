# 修改caffemodel文件方法
#### 首先是必须要做的，读取prototxt和caffemodel文件到caffe中，读取之后才可以对其进行操作(我按照jupter写的)
#### 1.导入caffe环境
```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
# configure plotting
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
```
#### 2.打开caffemodel和prototxt文件
```python
caffe.set_mode_cpu
net0 = caffe.Net('test23-4-24.prototxt',\
    'pvanet_frcnn_384_iter_5000.caffemodel',caffe.TEST) #TEST/TRAIN
conv1_w = net0.params['conv1_1/conv'][0].data
#模型参数都存在了net.params这个有序字典里，对这就是python里的那个字典，所以对模型参数的操作和对python字典操作一样。['conv1_1/conv']是键名，[0]是权的维度
```
#### 3.遍历所有的层名
```python
keys0 = net0.params.keys()
print net0.params.keys()
for key0 in keys0:   # 输出所有层名，参数
    print key0
    try:
        print net1.params[key1][0].data
    except IndexError:
        continue
    try:
        print net1.params[key1][1].data
    except IndexError:
        continue
    try:
        print net1.params[key1][2].data
    except IndexError:
        continue
    finally:
        print '\n'
```
