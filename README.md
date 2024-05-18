# RIGA at SMM4H-2024 Task 1: Enhancing ADE discovery with GPT-4

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The repository describes RIGA team submission to SMM4H-2024 Task 1

## Getting started
1. Create a new environment
    ```bash
    python -m venv venv
    ```
2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
3. Now your environment is ready. Next step is get the data from. You need to contact the [SMM4H-2024](https://healthlanguageprocessing.org/smm4h-2024/) organizers and request the data. Then put the data in `data` directory.

## Project structure
```
./scripts - Data preprocessing scripts
./train.py - script for trainig a classifier
./train_ner.py - script for trainig a NER model
./predict.py - script for running the inference of a classifier
./predict_ner.py - script for running the inference of a NER model
```
