# Low-No-code prototype with Pycaret & FastAPI


## Why the APP?

This app allows users to apply a variety of AI/ML tools without the need for coding. It integrates multiple frameworks designed specifically for AI/ML workflows, streamlining complex processes into a user-friendly interface. The app also includes predefined functions for data management, enabling users to easily handle data preparation, cleaning, and transformation tasks. Additionally, it offers robust visualization capabilities, allowing users to explore and interpret results effectively, all within a seamless, integrated environment.

## APP components 

The app is designed to connect shiny app with Pycaret library power. FastAPI help for etablish this connection.

## Setup
Prepare your Python environment (FastAPI + PyCaret)
Step 1: Create a new virtual environment
- Create a new virtual environment
```
python -m venv env1
```
- Load virtual environment created on Windows
```
env1\Scripts\activate 
```
- Load virtual environment created on Linux
```
source env1/bin/activate
```
Step 2: Install libraries needed
```
pip install fastapi uvicorn pycaret pandas python-multipart shap
```

The APP depends on some Python libraries :
- FastAPI : Create API easily and fastly
- Pycaret : Module for ML no-code
- Uvicorn : It help to launch FastAPI
- Python-multipart : 

To start the app, you would first need to have the package installed.

## Usage

Clone the repo first :)

### Python part
main.py file contains code to create an API with FastAPI and Pycaret.
Go to command line and run this following command to Launch FastAPI.
```
uvicorn main:app --reload
```

### R studio
Open app.R from R Studio and click to Run button. (Make sure you already have all packages)
Or also run the the following command to launch Shiny app :
```
shiny::runApp()
```
## License

- [GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007](./LICENSE)

