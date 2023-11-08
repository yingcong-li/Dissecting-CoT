import json
import os

from munch import Munch
import torch
import yaml

import models
from samplers import get_data_sampler
from tasks import get_task_sampler


def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = models.build_model(conf.model)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path)
        # hacky way of ensuring that if I'm trying to load keys that
        # aren't in the checkpoint, just use random
        for key in model.state_dict().keys():
            if key not in state["model_state_dict"].keys():
                state["model_state_dict"][key] = model.state_dict()[key]
        model.load_state_dict(state["model_state_dict"])
    elif step == 0:
        # return random init model
        return model, conf
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path)
        # hacky way of ensuring that if I'm trying to load keys that
        # aren't in the checkpoint, just use random
        for key in model.state_dict().keys():
            if key not in state_dict.keys():
                state_dict[key] = model.state_dict()[key]
        model.load_state_dict(state_dict)

    return model, conf


def eval_batch(model, task_sampler, xs, task_name=None):
    task = task_sampler()
    if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2"]:
        device = "cuda"
    else:
        device = "cpu"
    if task_name in ['relu_nn_regression']:
        ys, layer_activations = task.evaluate(xs)
        layer_activations = [act.to(device) for act in layer_activations]
    else:
        raise NotImplementedError

    if model.name.split("_")[0] in ["gpt2"]:
        pred = model.predict(xs.to(device), ys.to(device), layer_activations=layer_activations).detach()
    else:
        raise NotImplementedError
    metrics = task.get_metric()(pred.cpu(), ys)

    return metrics


def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}


def eval_model(
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    num_eval_examples=1280,
    batch_size=64,
    task_sampler_kwargs={},
):
    
    assert num_eval_examples % batch_size == 0
    data_sampler = get_data_sampler(data_name, n_dims)
    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, **task_sampler_kwargs
    )

    all_metrics = []
    for i in range(num_eval_examples // batch_size):
        xs = data_sampler.sample_xs(n_points, batch_size)
        metrics = eval_batch(model, task_sampler, xs, task_name)
        all_metrics.append(metrics)

    metrics = torch.cat(all_metrics, dim=0)

    return aggregate_metrics(metrics)


def build_evals(conf):
    n_dims = conf.model.n_dims
    n_points = conf.training.curriculum.points.end
    batch_size = conf.training.batch_size

    task_name = conf.training.task
    data_name = conf.training.data

    evaluation_kwarg = {
        "task_name": task_name,
        "task_sampler_kwargs": conf.training.task_kwargs,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
    }

    return evaluation_kwarg


def compute_evals(model, evaluation_kwargs, save_path=None):
    metrics = eval_model(model, **evaluation_kwargs)

    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(metrics, fp, indent=2)

    return metrics


def get_run_metrics(run_path, step=-1, cache=True):
    model, conf = get_model_from_run(run_path, step)
    model = model.cuda().eval()
    evaluation_kwargs = build_evals(conf)

    if not cache:
        save_path = None
    elif step == -1:
        save_path = os.path.join(run_path, "metrics.json")
    else:
        save_path = os.path.join(run_path, f"metrics_{step}.json")

    try:
        with open(save_path) as fp:
            metrics = json.load(fp)
            return metrics
    except Exception:
        metrics = {}

    metrics = compute_evals(model, evaluation_kwargs, save_path)
    return metrics



