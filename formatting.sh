#!/bin/sh
black . -v --exclude="/networkx-metis/"
flake8 . --count --ignore="E501 W503 F841 E203" --show-source --statistics --exclude=./.*,networkx-metis
