# 修改caffemodel文件方法（完成）
### ♥♥♥为caffemodel添加新层并赋值参数♥♥♥
### 下面这个例子是在caffemodel中增加层，并且为新层添加参数（这里我的新层直接复制的老层的参数，用于知识蒸馏的预训练模型）
#### 一、首先是必须要做的，读取prototxt和caffemodel文件到caffe中，读取之后才可以对其进行操作(我按照jupter写的)
#### 0.复制prototxt文件，在新prototxt文件中直接增加新层结构（必须）⭐
#### 1.导入caffe环境（必须）
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
#### 2.打开caffemodel和prototxt文件（必须）
```python
caffe.set_mode_cpu
#模型参数都存在了net.params这个有序字典里，对这就是python里的那个字典，所以对模型参数的操作和对python字典操作一样。['conv1_1/conv']是键名，[0]是权的维度
#老模型
model_def = '/home/omnisky/caffe/jobs/params/old.prototxt'
model_weights = '/home/omnisky/caffe/jobs/params/MSSC-64.caffemodel'

net_old = caffe.Net(model_def,
                model_weights,
                caffe.TEST)
#新模型                
caffe.set_device(2)
new_model_def = '/home/omnisky/caffe/jobs/params/new.prototxt'
new_model_weights = '/home/omnisky/caffe/jobs/params/MSSC-64.caffemodel'
net_new = caffe.Net(new_model_def,
                new_model_weights,
                caffe.TEST)
```
#### 3.遍历所有的层名（可选）
```python
keys0 = net0.params.keys()
print net0.params.keys()
for key0 in keys0:   # 输出所有层名，参数
    print key0
    try:
        print net1.params[key0][0].data
    except IndexError:
        continue
    try:
        print net1.params[key0][1].data
    except IndexError:
        continue
    try:
        print net1.params[key0][2].data
    except IndexError:
        continue
    finally:
        print '\n'
```
#### 4.参数复制(必须) 这里我把新添加的层的名字放在了txt里
```python
new_layers = []
with open('/home/omnisky/caffe/jobs/new_layers','r') as f:
    for line in f:
        new_layers.append(line)
        net_new_weights = net_new.params[line.strip('\n')][0].data
        print('this is old params ',line.strip('\n').replace('/student',''))
        net_old_weights = net_old.params[line.strip('\n').replace('/student','')][0].data
        #print net_old_weights
     #赋值   
        net_new.params[line.strip('\n')][0].data.flat = net_old_weights
        print('this is new params ',line.strip('\n'))
        print net_new_weights
#查看权重
net_old_weights = net_old.params['stage4_tb/ext/pm2/b2c/scale'][0].data
print net_old_weights
```
