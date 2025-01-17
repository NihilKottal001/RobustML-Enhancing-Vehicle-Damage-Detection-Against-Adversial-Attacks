[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/LU8t0ikG)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=17561560&assignment_repo_type=AssignmentRepo)
# RUC Practical AI - Final Project: : "RobustML - Enhancing Vehicle Damage Detection Against Adversarial Attacks"

Purpose

This project is designed to explore and demonstrate the vulnerability of machine learning models used by insurance companies to adversarial attacks. Specifically, it focuses on models that assess vehicle damage through image analysis. The goal is to show how these attacks could potentially lead to erroneous damage assessments, thereby affecting the accuracy of insurance claims processing. This is crucial for developing strategies to enhance the robustness of such systems against malicious manipulations.

Usage Instructions

* Refer requirements.txt for more information on the libraries being used
* Make sure the libraries have been installed before running the project
* The data has been augmented before being used in the project. For more information on the data augmentation part please refer the “Data Augmentation” section of “RobustML.ipynb”.
* Once all the libraries are installed, run the “RobustML.ipynb” to view the insights and results. 

File definitions

* archive.zip - original data file 
* requirements.txt - required libraries to be installed
* data_creation.py - This is a python file containing the code to restructure the data in a way such that it can be easily segregated into labels and data and then loaded into the model.
* adversial_utils.py - This python file includes the class handling the functions facilitating us in attacking and defending the model
* architecture.py - This python file includes the architecture of our neural network
* ImageDataset.py - This python file includes the class helping us in creating dataloaders which align with our neural network architecture
* training_utils.py - This python file contains functions which are used to train and test our neural network
* visualization.py - This python file contains functions which help us in visualizations used in the notebook
* RobustML.ipynb - Main notebook containing the insights and the results

Known Issues

* The model may incorrectly classify minor scratches as medium damage under certain lighting conditions.
* Adversarial robustness is currently limited to specific types of attacks such as FGSM.

Feature Roadmap

* Integration of Additional Adversarial Attack Scenarios: Future versions will include more sophisticated adversarial attacks like DeepFool and PGD.
* Enhancement of Damage Assessment Accuracy: Plans to incorporate more granular damage categories and improve low-light performance.
* Deployment as a Web Service: Enable real-time damage assessment through a web application interface.

Contributing

Contributors are welcome to enhance the functionalities of this project. If you are interested in contributing, please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change. Ensure to update tests as appropriate.

Contact
For any inquiries, please reach out via email at nihil.kottal@rutgers.edu
