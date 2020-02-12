import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, Trainer
from mxnet.gluon.data import DataLoader, ArrayDataset

mx.random.seed(12345)  # Added for reproducibility

# 生成数据
def get_random_data(size, ctx):
    x = nd.normal(0, 1, shape=(size, 10), ctx=ctx)
    y = x.sum(axis=1) > 3
    return x, y

ctx = mx.cpu()
train_data_size = 1000
val_data_size = 100
batch_size = 10

# 处理数据
train_x, train_ground_truth_class = get_random_data(train_data_size, ctx)
train_dataset = ArrayDataset(train_x, train_ground_truth_class)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_x, val_ground_truth_class = get_random_data(val_data_size, ctx)
val_dataset = ArrayDataset(val_x, val_ground_truth_class)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# 定义和训练模型
net = nn.HybridSequential()

with net.name_scope():
    net.add(nn.Dense(units=10, activation='relu'))  # input layer
    net.add(nn.Dense(units=10, activation='relu'))   # inner layer 1
    net.add(nn.Dense(units=10, activation='relu'))   # inner layer 2
    net.add(nn.Dense(units=1))   # output layer: notice, it must have only 1 neuron

net.initialize(mx.init.Xavier())

loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
trainer = Trainer(params=net.collect_params(), optimizer='sgd',
                  optimizer_params={'learning_rate': 0.1})
accuracy = mx.metric.Accuracy()
f1 = mx.metric.F1()

def train_model():
    cumulative_train_loss = 0

    for i, (data, label) in enumerate(train_dataloader):
        with autograd.record():
            # Do forward pass on a batch of training data
            output = net(data)

            # Calculate loss for the training data batch
            loss_result = loss(output, label)

        # Calculate gradients
        loss_result.backward()

        # Update parameters of the network
        trainer.step(batch_size)

        # sum losses of every batch
        cumulative_train_loss += nd.sum(loss_result).asscalar()

    return cumulative_train_loss

# 验证模型
def validate_model(threshold):
    cumulative_val_loss = 0

    for i, (val_data, val_ground_truth_class) in enumerate(val_dataloader):
        # Do forward pass on a batch of validation data
        output = net(val_data)

        # Similar to cumulative training loss, calculate cumulative validation loss
        cumulative_val_loss += nd.sum(loss(output, val_ground_truth_class)).asscalar()

        # getting prediction as a sigmoid
        prediction = net(val_data).sigmoid()

        # Converting neuron outputs to classes
        predicted_classes = mx.nd.ceil(prediction - threshold)

        # Update validation accuracy
        accuracy.update(val_ground_truth_class, predicted_classes.reshape(-1))

        # calculate probabilities of belonging to different classes. F1 metric works only with this notation
        prediction = prediction.reshape(-1)
        probabilities = mx.nd.stack(1 - prediction, prediction, axis=1)

        f1.update(val_ground_truth_class, probabilities)

    return cumulative_val_loss

epochs = 10
threshold = 0.5

for e in range(epochs):
    avg_train_loss = train_model() / train_data_size
    avg_val_loss = validate_model(threshold) / val_data_size

    print("Epoch: %s, Training loss: %.2f, Validation loss: %.2f, Validation accuracy: %.2f, F1 score: %.2f" %
          (e, avg_train_loss, avg_val_loss, accuracy.get()[1], f1.get()[1]))

    # we reset accuracy, so the new epoch's accuracy would be calculated from the blank state
    accuracy.reset()