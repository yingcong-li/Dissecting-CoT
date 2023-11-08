import os
import shutil
from random import randint
import uuid
import datetime

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model



import wandb

torch.backends.cudnn.benchmark = True



def train_step(model, xs, ys, optimizer, loss_func, layer_activations=None):
    optimizer.zero_grad()
    losses, loss = model(xs, ys, loss_func, layer_activations=layer_activations)
    loss.backward()
    optimizer.step()
    losses = [ls.detach().item() for ls in losses]
    return losses, loss.detach().item()


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    data_sampler_args = {}
    task_sampler_args = {}

    task = task_sampler(**task_sampler_args)

    for i in pbar:
        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )

        loss_func = task.get_training_metric()
        if args.training.task in ['relu_nn_regression']:
            task = task_sampler(**task_sampler_args)
            ys, layer_activations = task.evaluate(xs)
            layer_activations = [act.cuda() for act in layer_activations]
            losses, loss = train_step(model, xs.cuda(), ys.cuda(), optimizer,
                loss_func, layer_activations=layer_activations)
        else:
            raise NotImplementedError

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "stepwise/loss": dict(
                        zip(list(range(len(losses))), losses)
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ) or (i == args.training.train_steps - 1):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            mode="disabled" if args.debug_mode else "online",
            resume=True,
        )

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)

    if args.debug_mode:
        # delete wandb directory when done
        print("Deleting out_dir {} because of debug mode".format(args.out_dir))
        shutil.rmtree("{}".format(args.out_dir), ignore_errors=True)


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2_nn"]
    print(f"Running with: {args}")

    if args.debug_mode:
        args.out_dir = "../models/debug"

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir
        # add a timestamp here
        args.wandb['timestamp'] = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
