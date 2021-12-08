[![Codacy Badge](https://app.codacy.com/project/badge/Grade/f8bcca65fd434829ba9ca3349ce37966)](https://www.codacy.com/gh/Pugavkomm/-test-multy_cognitive_tasks/dashboard?utm_source=github.com&utm_medium=referral&utm_content=Pugavkomm/-test-multy_cognitive_tasks&utm_campaign=Badge_Grade)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# cgtasknet

## Table of Contents

[About](#About)  
[Requirements](#Requirements)  
[Tasks](#Tasks)

## About

A library aimed at studying the dynamics of spiking neural networks while solving various cognitive tasks:

1. Romo task;
2. Context decision-making;
3. Antisaccade;
4. Pro-saccade;
5. etc;

---

## Requirements

Main dependencies:

1. [torch](https://pytorch.org/), [norse](https://github.com/norse/norse) -- Model and learning;
1. [numpy](https://numpy.org/) -- Prepare datasets;
1. [matplotlib](https://matplotlib.org/) -- Data visualization.

```
absl-py==0.15.0
aiohttp>=3.7.4
async-timeout==3.0.1
attrs==21.2.0
backcall==0.2.0
cachetools==4.2.4
certifi==2021.10.8
chardet==4.0.0
charset-normalizer==2.0.7
cloudpickle==2.0.0
cycler==0.11.0
debugpy==1.5.1
decorator==5.1.0
entrypoints==0.3
fsspec==2021.10.1
future==0.18.2
google-auth==2.3.2
google-auth-oauthlib==0.4.6
grpcio==1.41.1
idna==3.3
ipykernel==6.4.2
ipython==7.29.0
ipython-genutils==0.2.0
jedi==0.18.0
jupyter-client==7.0.6
jupyter-core==4.9.1
kiwisolver==1.3.2
Markdown==3.3.4
matplotlib==3.4.3
matplotlib-inline==0.1.3
multidict==5.2.0
nest-asyncio==1.5.1
norse==0.0.7.post1
numpy==1.21.3
oauthlib==3.1.1
packaging==21.2
parso==0.8.2
pexpect==4.8.0
pickleshare==0.7.5
Pillow==8.4.0
prompt-toolkit==3.0.21
protobuf==3.19.1
ptyprocess==0.7.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pyDeprecate==0.3.1
Pygments==2.10.0
pyparsing==2.4.7
python-dateutil==2.8.2
pytorch-lightning==1.4.9
PyYAML==6.0
pyzmq==22.3.0
requests==2.26.0
requests-oauthlib==1.3.0
rsa==4.7.2
scipy==1.7.1
six==1.16.0
tensorboard==2.7.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
tk==0.1.0
torch==1.10.0
torchmetrics==0.6.0
torchvision==0.11.1
tornado==6.1
tqdm==4.62.3
traitlets==5.1.1
typing-extensions==3.10.0.2
urllib3==1.26.7
wcwidth==0.2.5
Werkzeug==2.0.2
yarl==1.7.0
```

## Tasks

Several classes of cognitive tasks are considered:

1. [Romo task](https://www.nature.com/articles/20939)
1. [Context decision making task](https://www.nature.com/articles/nature12742)
1. ...
