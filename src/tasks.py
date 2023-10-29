import math
import torch


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(task_name, n_dims, batch_size, **kwargs):
    task_names_to_classes = {
        "relu_nn_regression": ReluNNRegression,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        return task_cls(n_dims, batch_size, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class ReluNNRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        hidden_layer_size=4,
        n_layers=2,
        mode='relu'
    ):
        super(ReluNNRegression, self).__init__(n_dims, batch_size)
        self.hidden_layer_size = hidden_layer_size
        self.n_layers = n_layers
        if n_layers < 2:
            raise ValueError("Number of layers should not be smaller than 2.")

        self.W_init = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
        self.Ws = torch.randn(self.n_layers-2, self.b_size, hidden_layer_size, hidden_layer_size)
        self.v = torch.randn(self.b_size, hidden_layer_size, 1)

        if mode == 'relu':
            self.act_func = torch.nn.ReLU()
        elif mode == 'tanh':
            self.act_func = torch.nn.Tanh()
        else:
            print("Unknown activation function")
            raise NotImplementedError


    def evaluate(self, xs_b):
        W_init = self.W_init.to(xs_b.device)
        Ws = self.Ws.to(xs_b.device)
        v = self.v.to(xs_b.device)

        activ = self.act_func(xs_b @ W_init) * math.sqrt(2 / self.hidden_layer_size)
        layer_activations = [activ]
        for i in range(self.n_layers-2):
            activ = self.act_func(activ @ Ws[i]) * math.sqrt(2 / self.hidden_layer_size)
            layer_activations.append(activ)
        ys_b_nn = (activ @ v)[:, :, 0]        
        return ys_b_nn, layer_activations

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return squared_error

