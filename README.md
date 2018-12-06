# tensor-ocr

Optical character recognition using Tensorflow.

## How to install

You need a python 3.6 installation, because newer versions don't have support for tensorflow.

Clone the directory:

```git clone https://github.com/jsmolka/tensor-ocr```

or download a release.

Download model.zip from https://github.com/jsmolka/tensor-ocr/releases and extract its contents into the src/data directory of the downloaded directory.

Change into the downloaded directory and install the required python packages:

```pip install -r requirements.txt```


## How to use

To start the GUI use:

```python src/app.pyw```

For training and testing use: 

```python src/main.py```

There are different command line parameters available:

```python src/main.py --analyse``` Get Parameters of the IAM dataset

```python src/main.py --convert``` Preprocess IAM dataset

```python src/main.py --rotate``` Creates more samples by rotating them

```python src/main.py --train``` Train the model

```python src/main.py --test``` Test the model