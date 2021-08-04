"""Utilities to create and store records of jobs."""
import datetime
import json
from os import environ
from pathlib import Path
import platform
import sys
from typing import Any, Dict, Optional

import click

import git
import gray_zone

from numpy.random import randint, seed


def is_slurm_job() -> bool:
    """Determine whether the current process is running in a slurm cluster.

    Returns
    -------
    bool
        True if the current process is running in a slurm job

    """
    return 'SLURM_JOB_ID' in environ


def get_job_record(random_seed: Optional[int] = None) -> Dict[str, Any]:
    """Get a record of the current cluster environment.

    A cluster record is a dictionary of information about the current cluster
    environment. It is intended to be stored to file as a record of a job
    running.

    The following information is included:
    - slurm_job_id
    - name of the slurm log file
    - username that submitted the job
    - date and time the job was run
    - git commit hash of code version used
    - hostname on which the job was allocated
    - list of GPUs that were available to the job

    Parameters
    ----------
    random_seed: int
        Specifies the random seed used in this job.
        If not specified, generate a random seed to use with this job.

    Returns
    -------
    dict
        Dictionary containing the relevant information.

    """
    # Generate a random seed if needed
    if random_seed is None:
        random_seed = randint(1000000)

    # Set up
    seed(random_seed)  # numpy

    record = {}
    record['RANDOM_SEED'] = random_seed

    # Store the entire argument list
    record['ARGV'] = sys.argv

    # Store the time this command was run
    record['EXECUTION_TIME'] = str(datetime.datetime.now())

    # Get the current git commit, branch and state
    repo = git.Repo(gray_zone.__path__[0], search_parent_directories=True)  # type: ignore  # mypy issue #1422
    try:
        record['GIT_ACTIVE_BRANCH'] = str(repo.active_branch)
    except TypeError:
        # We are in detached state
        record['GIT_ACTIVE_BRANCH'] = 'DETACHED'
    record['GIT_COMMIT_HASH'] = str(repo.head.commit)
    record['GIT_IS_DIRTY'] = str(repo.is_dirty())

    # Get the machine's hostname
    record['NODE'] = platform.node()

    # A list of environment variables to store
    var_list = [
        'SLURM_JOB_USER',
        'SLURM_JOB_ID',
        'SLURM_NODELIST',
        'CUDA_VISIBLE_DEVICES',
        'SLURM_NPROCS',
        'SLURM_JOB_SCRATCHDIR',
        'SLURM_CPUS_ON_NODE',
        'USER',
    ]

    for var in var_list:
        if var in environ:
            record[var] = environ[var]
        else:
            record[var] = ''

    return record


def save_job_record(
    output_dir: Path,
    name: str = 'job_record.json',
    record: Optional[Dict[str, Any]] = None
) -> None:
    """Save a job record file into an existing output directory.

    Parameters
    ----------
    output_dir: Path
        Path to the output_directory
    name: str
        Name of the job record file, defaults to job_record.json
    record: Optional[Dict[str: Any]]
        A job record, as returned by get_job_record. If None is provided
        get_job_record will be called to create one.

    """
    # Get a record if none was provided
    if record is None:
        record = get_job_record()

    # Save the record to file
    if not name.endswith('.json'):
        name += '.json'
    job_record_path = Path(output_dir) / name
    with job_record_path.open('w') as jf:
        json.dump(record, jf, indent=4)


@click.command()
def test_job_record() -> None:
    """Test function to create a job record in the current directory."""
    save_job_record(Path('.'))
