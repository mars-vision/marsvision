#!/bin/bash
mypy -p marsvision
pytest --cov-report term-missing --cov=marsvision test
cd docs
run_build.bat
read -n 1 -s -r -p "Press any key to continue"