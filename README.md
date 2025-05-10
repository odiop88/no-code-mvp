# Low-/No- code prototype with Pycaret


## Why the APP?

This app allows users to apply a variety of AI/ML tools without the need for coding. It integrates multiple frameworks designed specifically for AI/ML workflows, streamlining complex processes into a user-friendly interface. The app also includes predefined functions for data management, enabling users to easily handle data preparation, cleaning, and transformation tasks. Additionally, it offers robust visualization capabilities, allowing users to explore and interpret results effectively, all within a seamless, integrated environment.

## APP components 

The app is designed to connect shiny app with Pycaret library

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
pip install fastapi uvicorn pycaret pandas python-multipart
```

The APP depends on some Python libraries :
- FastAPI : Create API easily and fastly
- Pycaret : Module for ML no-code
- Uvicorn : It help to launch FastAPI
- Python-multipart : 

To start the app, you would first need to have the package installed.

Step 3: Launch FastAPI
Go to command line and run this following command
```
uvicorn main:app --reload
```
Step 4:  Launch Shiny App
Go to app.R file and run it (Make sure you already have all packages)

## Usage

Clone the repo first :)

### R studio

Open app.R from R Studio and click to Run button.

### Python part

main.py file contains code to create an API with FastAPI and Pycaret

## License

- [GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007](./LICENSE)

