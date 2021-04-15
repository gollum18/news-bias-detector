# News-Bias-Detector
Implements news bias detection from the All the News dataset from Kaggle. 
Classification is performed on publication labels obtained from mediabiasfactcheck.com.


Classification is performed utilizing the following fully-configurable techniques:
- AdaBoost Ensemble Classification
- Gradient Boosting Ensemble Classification
- Extremely Random Trees Enseble Classification
- PyTorch Neural Network Model


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