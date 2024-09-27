#!/usr/bin/env bash

# This script will build the package and updates dependencies by updateing the pyproject.toml file  of the MetObs toolkit

echo " ---- Building and Updating Metobs Toolkit ----"

DEPLOY_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ${DEPLOY_DIR}
cd .. #Navigate to workdir
WORKDIR=$(pwd)

DISTDIR=${WORKDIR}/dist


#1. cleanup previous builds
rm ${DISTDIR}/*.whl
rm ${DISTDIR}/*.tar.gz

#2. Update the dependencies in the  toml
poetry update

# Toolkit dependencies
poetry add pandas@latest
poetry add yfinance@latest
poetry add matplotlib@latest


# Toolkit DEV group
poetry add poetry@latest --group dev


poetry install --all-extras
poetry show



poetry build
cd ${DEPLOY_DIR}
