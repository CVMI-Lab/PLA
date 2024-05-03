#### Requirements
All the codes are tested in the following environment:
- Python 3.7+
- PyTorch 1.8
- CUDA 11.1
- [spconv v2.x](https://github.com/traveller59/spconv)

#### Install dependent libraries
a. Clone this repository.
```bash
git clone https://github.com/CVMI-Lab/PLA.git
git fetch -all
git checkout regionplc
```

b. Install the dependent libraries as follows:

* Install the dependent Python libraries (Please note that you need to install the correct version of `torch` and `spconv` according to your CUDA version): 
    ```bash
    pip install -r requirements.txt 
    ```

* Install [SoftGroup](https://github.com/thangvubk/SoftGroup) following its [official guidance](https://github.com/thangvubk/SoftGroup/blob/main/docs/installation.md).
    ```bash
    cd pcseg/external_libs/softgroup_ops
    python3 setup.py build_ext develop
    cd ../../..
    ```

* Install [pcseg](../pcseg)
    ```bash
    python3 setup.py develop
    ```
