import paddle
import paddle.nn.functional as F
import numpy as np
import dataset
import network

def evaluation(opt):
    paddle.set_device('gpu:0') if opt.use_gpu else paddle.set_device('cpu')

    print('start evaluation .......')

    model = network.LeNet(opt)
    model_state_dict = paddle.load(opt.params_file_path)
    model.load_dict(model_state_dict)
    model.eval()
    eval_loader = dataset.data_loader(opt)

    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(eval_loader()):
        x_data, y_data = data
        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)
        y_data = y_data.astype(np.int64)
        label_64 = paddle.to_tensor(y_data)

        prediction, acc = model(img, label_64)

        loss = F.binary_cross_entropy_with_logits(prediction, label)
        avg_loss = paddle.mean(loss)

        if batch_id % 1 == 0:
            print("batch_id: {}, loss is: {:.4f}".format(batch_id, float(avg_loss.numpy())))

        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))

    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    print('loss={:.4f}, acc={:.4f}'.format(avg_loss_val_mean, acc_val_mean))

