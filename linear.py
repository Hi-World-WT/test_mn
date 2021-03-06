import mxnet as mx
from mxnet import autograd, nd, gluon, init
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss

#================生成数据
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
ctx = mx.cpu()
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
 
#=================读取数据
batch_size=10
dataset=gdata.ArrayDataset(features,labels)
data_iter=gdata.DataLoader(dataset,batch_size,shuffle=True)
 
#======================定义模型
net = nn.Sequential()#这一开始是空的，定义的容器
net.collect_params().initialize(init.Xavier(), ctx = ctx)
net.add(nn.Dense(1))
 
# =========================初始化模型参数
net.initialize(init.Normal(sigma=0.01))
 
#====================定义损失函数
loss=gloss.L2Loss()#平方损失又称L2范数损失
 
#============定义优化函数
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})
 
#===============训练模型
num_epochs=3
for epoch in range(1,num_epochs+1):
    for X,y in data_iter:
        with autograd.record():
            l=loss(net(X),y)
        l.backward()
        trainer.step(batch_size)
    l=loss(net(features),labels)
    print('epoch %d,loss: %f'%(epoch,l.mean().asnumpy()))

