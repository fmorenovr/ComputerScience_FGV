## Installation

First, download data:

```
bash get_data.sh
```

Then, copy models and code:

```
bash get_code.sh
```

Next, install conda environment:

```
    conda env create --name server_styleGAN --file requirements.yaml 
    conda env export > requirements.yaml
```

or using pip:

```
    python -m venv server_styleGAN
    source server_styleGAN/bin/activate
    pip list --format=freeze > requirements.txt
    python -m pip install -r requirements.txt 
```
