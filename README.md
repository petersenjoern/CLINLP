# CLINLP
Parsing Clinical and other biomedical data the smart way 

## 1. Download ctgov data and spin-up in a postgres db in docker
```bash
./ctgov <yyyymm01>
```

## 2. Fill in the Named Entity Recoginition configuration yaml
```bash
vi ./src/ner/config_ner.yaml
```

## 3. Run Named Entity Recognition by using Hydra CLI:
```bash
./src/ner/ner.py
```


## Training with a GPU:

1. you may want to consider to reduce the total power consumption, and thereby reduce the vRAM may temp. To find the ideal configuration, observe your vRAM under heavy GPU load. Tooling on Linux is not good for doing so. I suggest you use windows HWinfo64
```bash
sudo nvidia-smi -i 0 -pl 250
watch -n 1 nvidia-smi
```bash

