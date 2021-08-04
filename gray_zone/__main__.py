"""Entrypoint. Define and run CLI."""
import click

from gray_zone.run_model import run_model
from gray_zone.records import test_job_record
from gray_zone.train.train import train


@click.group()
def cli() -> None:
    """CLI group to which all specific entrypoints will be added."""
    pass


cli.add_command(run_model)
cli.add_command(train)
cli.add_command(test_job_record)


if __name__ == '__main__':
    cli()
