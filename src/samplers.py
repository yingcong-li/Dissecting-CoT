import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


class GaussianSampler(DataSampler):
    def __init__(self, n_dims):
        super().__init__(n_dims)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None):
        xs_b = torch.randn(b_size, n_points, self.n_dims)
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b
