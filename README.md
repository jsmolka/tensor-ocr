# tensor-ocr

Optical character recognition using Tensorflow.

## Setup
1. Install Python 3.6 or lower (Tensorflow does not support newer versions).
2. Clone or download the repository.
3. Install the requirements using ```pip install -r requirements.txt```.
4. Download a pretrained model from the [release page](https://github.com/jsmolka/tensor-ocr/releases) if you do not want to train the model yourself.
5. Extract the archive and move the `model.json` and `weights.h5` files into the `data` directory.

## Training
If you want to train a model yourself you need to download the [IAM dataset](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/iam-handwriting-database). After that you need to convert the dataset and train the model using the commands from the section below.

## Commands
| Command                        | Action                                                           |
| ------------------------------ | ---------------------------------------------------------------- |
| ```python app.pyw```           | Starts the GUI application.                                      |
| ```python main.py --analyse``` | Analyses the IAM dataset.                                        |
| ```python main.py --convert``` | Converts the IAM dataset into the desired format.                |
| ```python main.py --rotate```  | Rotates the converted IAM dataset to create more training data.  |
| ```python main.py --train```   | Trains the model using the converted IAM dataset.                |
| ```python main.py --test```    | Tests the trained model.                                         |
