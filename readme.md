# News-Bias-Detector
Implements news bias detection from the All the News dataset from Kaggle. 
Classification is performed on publication labels obtained from mediabiasfactcheck.com.


Classification is performed utilizing the following fully-configurable techniques:
- AdaBoost Ensemble Classification
- Gradient Boosting Ensemble Classification
- Extremely Random Trees Ensemble Classification
- PyTorch Neural Network Model


Note that the PyTorch model found in `mbd/model.py` currently achieves approximately 52% accuracy with the current feature set after 100 training epochs. This model started of far more complex than the model currently found in the aforementioned module. After three iterations of reducing it's complexity, I was able to get the accuracy from ~30% to what you see now. It is clear to me that this model requires additional work to be considered a successful classification model. This project has shown that the feature set I have chosen does work to uniquely identify class instances. Further reifinement to the neural network model is to be considered future work.


The `data` folder only currently contains generated features as the features file itself is ~88MB in size. This repository contains code for generating the feature set yourself, but it is space expensive to do so, requiring upwards of ~12GB of RAM. `__init__.py` contains *some of the code* needed to generate the features but it is incomplete. If you want to generate the features yourself, you will need to string together the functions I use to do so from the other modules in the repository.


Documentation of all functions is forthcoming.


## How-To
To run the classifiers do the following:
1. Open a terminal or PowerShell instance.
2. Navigate to the directory containing this project.
3. Create a Python virtual environment `python3 -m venv venv` on Linux or `python -m venv venv` on Windows.
4. Activate the virtual environment `. venv/bin/activate` on Linux or `.\venv\Scripts\activate` on Windows.
5. Install required libraries: `pip install -r requirements.txt`.
6. Start one of the runner scripts: `python mbd/model.py <ensemble|nn>`


Include the `--help` tag will print out automatically generated help text to the console.


When running the `model.py` script, you must include either `ensemble` or `nn` as an argument to the script. `ensemble` will run the ensemble classification methods while `nn` will run the PyTorch neural network.