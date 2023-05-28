# Installation

### Installation

**1) Installing dependencies using conda**

```
    conda env create --name server_CPCA --file requirements.yaml 
    conda env export > requirements.yaml
```

**2) Installing using pyenv**

```
    python3 -m venv server_CPCA
    source server_CPCA/bin/activate
    pip list --format=freeze > requirements.txt
    python3 -m pip install -r requirements.txt 
```
