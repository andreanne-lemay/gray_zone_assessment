#!/bin/bash
set -e

# Edit these
REPO_DEPLOY_TOKEN_USERNAME="deploy_token_1"
REPO_DEPLOY_TOKEN_PASSWORD="ABC123"

# Clone the main project repo and install the python package
git clone https://{REPO_DEPLOY_TOKEN_USERNAME}:{REPO_DEPLOY_TOKEN_PASSWORD}@gitlab.ccds.io/open-source/ml_proj_template.git \
	${SLURM_JOB_SCRATCHDIR}/ml_proj
cd ${SLURM_JOB_SCRATCHDIR}/ml_proj

# Check out a specific commit according to the environment variable ENTRYPOINT_COMMIT
if [[ -v ENTRYPOINT_COMMIT ]]; then
	git checkout $ENTRYPOINT_COMMIT
fi
pip install --no-dependencies -e .
cd ..

# Configure minio client
mc config host add ccds https://ogw.ccds.io $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY

# Kick off the main python entrypoint
python -m ml_proj "$@"
