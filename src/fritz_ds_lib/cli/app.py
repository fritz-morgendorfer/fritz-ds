import click
import yaml

from fritz_ds_lib.cli.config import AppConfig
from fritz_ds_lib.core.names import DatasetType
from fritz_ds_lib.job import cv, evaluate, predict, train


@click.group()
@click.option(
    "--cfg",
    help="Path to the config file.",
    type=click.Path(exists=True),
    default=None,
)
@click.pass_context
def cli(ctx, cfg):
    """Entrypoint of the click application."""
    with open(cfg, "r") as fin:
        cfg = yaml.safe_load(fin)
    ctx.obj = AppConfig(**cfg)


@cli.command("train")
@click.pass_obj
def cli_train(cfg: AppConfig) -> None:
    train.train(cfg)


@cli.command("predict")
@click.option(
    "--dataset",
    type=click.Choice(["validation", "test"]),
    default="test",
)
@click.pass_obj
def cli_predict(cfg: AppConfig, dataset: DatasetType) -> None:
    predict.predict(cfg, dataset)


@cli.command("evaluate")
@click.option(
    "--dataset",
    type=click.Choice(["validation", "test"]),
    default="test",
)
@click.pass_obj
def cli_evaluate(cfg: AppConfig, dataset: DatasetType) -> None:
    evaluate.evaluate(cfg, dataset)


@cli.command("cv")
@click.pass_obj
def cli_cv(cfg: AppConfig) -> None:
    cv.cv(cfg)


if __name__ == '__main__':
    cli()
