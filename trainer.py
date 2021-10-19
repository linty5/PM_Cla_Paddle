import paddle
import paddle.nn.functional as F
import numpy as np
import dataset
import network

def train_pm(opt):
    paddle.set_device('gpu:0') if opt.use_gpu else paddle.set_device('cpu')

    print('start training ... ')

    model = network.LeNet(opt)
    optimizer = paddle.optimizer.Momentum(opt.lr, opt.momentum, parameters=model.parameters())
    model.train()
    train_loader = dataset.data_loader(opt)
    valid_loader = dataset.valid_data_loader(opt)

    for epoch in range(opt.epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)

            logits = model(img)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            avg_loss = paddle.mean(loss)

            if batch_id % 1 == 0:
                print("epoch: {}, batch_id: {}, loss is: {:.4f}".format(epoch, batch_id, float(avg_loss.numpy())))

            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            logits = model(img)

            pred = F.sigmoid(logits)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            pred2 = pred * (-1.0) + 1.0
            pred = paddle.concat([pred2, pred], axis=1)
            acc = paddle.metric.accuracy(pred, paddle.cast(label, dtype='int64'))

            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print("[validation] accuracy/loss: {:.4f}/{:.4f}".format(np.mean(accuracies), np.mean(losses)))
        model.train()

        paddle.save(model.state_dict(), 'palm.pdparams')
        paddle.save(optimizer.state_dict(), 'palm.pdopt')