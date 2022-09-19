[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyTest](https://github.com/Pugavkomm/cgtasknet/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/Pugavkomm/cgtasknet/actions/workflows/main.yml)

# cgtasknet

## Table of Contents

[About](#About)  
[Installation](#Installation)
[Requirements](#Requirements)  
[Tasks](#Tasks)
[TODO](#TODO)

## About

A library aimed at studying the dynamics of spiking neural networks while solving various cognitive tasks:

1. Romo task;
2. Decision-making task;
3. Context decision-making task;
4. Antisaccade task;
5. Pro-saccade task;
6. Go tasks family.

---

## Installation

### Installation from source code

***

1. `git clone https://github.com/Pugavkomm/cgtasknet.git`
2. You shold install [pytorch](https://pytorch.org/get-started/locally/) (stable version)
3. `python setup.py install`

> You can create your environment and install all the libraries you need to create networks

1. `git clone https://github.com/Pugavkomm/cgtasknet.git`
2. Create env: `python3 -m venv env`
3. Activate env: linux: `source env/bin/activate` | windows: `.\env\Scripts\activate`
4. You shold install [pytorch](https://pytorch.org/get-started/locally/) (stable version)
5. `pip3 install -r requirements.txt`
6. `python setup.py install`

### Installation from pypi

> TODO

### Docker

> TODO

## Requirements

Main dependencies:

1. [torch](https://pytorch.org/), [norse](https://github.com/norse/norse) -- Model and learning;
1. [numpy](https://numpy.org/) -- Prepare datasets;
1. [matplotlib](https://matplotlib.org/) -- Data visualization (Currently not in use!);

## Tasks

Several classes of cognitive tasks are considered:

1. [Romo task](https://www.nature.com/articles/20939)
2. [Context decision making task](https://www.nature.com/articles/nature12742)
3. [decision making task](...)
4. [Working memory tasks](...): DM with delay (just romo task), context DM with delay.
5. [Go/No-Go](...)

## TODO

* **Add my own wrapper over the loading model to load parameters (and save) as well;**
* ~~Add some tests for instrument_subgroups;~~
* ~~Add somet tests for instrument_pca;~~
* ~~Add some tests for all models.~~
