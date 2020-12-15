Semi-supervised User Geolocation Classification using Filtered User Relations and Location Regularizer on COVID Data
=================================================================


Introduction
------------
base_model branch: 
Contains instructions for running our data on original code
Primary difference is the new experiment script


Geolocation Datasets
--------------------
Datasets will be provided to the Prof. and TA via email, as we can't publically upload tweet content. 

Quick Start
-----------

1. Download the dataset .total file and place in ''./datasets''

```
.
├── datasets
│   ├── covid
│   │   ├── user_info.dev.gz
│   │   ├── user_info.test.gz
│   │   ├── user_info.train.gz
│   ├── dataset_world.total
│   ├── dataset_us.total
│   └── split.py
│  
├── data.py
├── data.pyc
├── deepcca.py
├── experiments
│   ├── cmu-concat-fractions.sh
│   ├── cmu-dcca-fractions.sh
│   ├── cmu-gcn-fractions.sh
│   ├── covid-gcn-fractions.sh
│   ├── cmu_layers_highway.sh
│   ├── cmu_layers_nohighway.sh
│   ├── feature_report.sh
│   ├── save_model.sh
│   └── tunebucket.sh
├── gcnmain.py
├── gcnmodel.py
├── gcnmodel.pyc
├── kdtree.py
├── kdtree.pyc
├── mlp.py
├── README.md
├── requirements.txt
└── utils.py

```

2. Create a new environment:

```conda create --name geo python=3.7```

Activate the environment:

```conda activate geo```

Install Libraries:

```pip install -r requirements.txt```

Upgrade Theano and Lasagne:

```
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip

pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

3. Create the data directory: 

```mkdir ./datasets/covid```

Split the total file into the data directory: 

```python split.py ./datasets/dataset_us.total ./datasets/covid```

GZIP all of the files in the data directory:

```
cd ./datasets/covid
gzip *
cd ../../
```

4. To run the experiments, look at the experiments directory.


COVID (runtime: 5-12):

```
    THEANO_FLAGS='device=cuda0,floatX=float32' nice -n 9 python -u gcnmain.py -hid 500 400 300 200 129 -bucket 50 -batch 500 -d ./datasets/covid/  -mindf 14 -reg 0.0 -dropout 0.0 -cel 5 -highway -seed $seed -lblfraction $fraction -builddata -silent #-maxdown 50
```
Run with: 
```sh ./experiments/covid-gcn-fractions.sh```

To switch back to the Master branch (that has the input word embeddings code modification): 
```git checkout master```
