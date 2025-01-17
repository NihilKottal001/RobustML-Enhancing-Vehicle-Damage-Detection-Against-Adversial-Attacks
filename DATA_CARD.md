# Data Card

Dataset Overview

* Name: Car Damage Assesment
* Source: https://www.kaggle.com/datasets/hamzamanssor/car-damage-assessment
* Author(s): Hamza Manssor
* Owner: Peltarion

 Purpose 
 
 This dataset serves as the foundation for a project aimed at assessing the impact of adversarial attacks on machine learning models utilized by insurance companies. The primary objective is to evaluate and demonstrate how these attacks can manipulate model predictions related to vehicle damage assessments. By understanding these vulnerabilities, the project seeks to illustrate potential risks such as inaccurate damage evaluations and fraudulent insurance claims. This insight is crucial for developing more robust machine learning systems that can resist adversarial manipulations, ensuring reliable and fair insurance processing

Data Pre Processing

The original dataset comprised eight distinct damage classes representing various types of vehicle damage:

- Head Lamp
- Door Scratch
- Glass Shatter
- Tail Lamp
- Bumper Dent
- Door Dent
- Bumper Scratch
- Unknown (used for damages that did not clearly fit into any of the other categories).

Simplification Process

To streamline the analysis and better align the dataset with the project's objectives of assessing damage intensity for insurance claims, the original eight classes were consolidated into four simplified categories:

* Good: Represents vehicles with no damage. This category was derived from images originally labeled as 'Unknown' that upon review, showed no visible damage.
* Low: Includes minor damages such as 'Door Scratch' and 'Bumper Scratch'.
* Medium: Encompasses moderate damages such as 'Door Dent', 'Head Lamp', and 'Tail Lamp'.
* Critical: Covers severe damages, indicative of significant vehicular harm. This category was derived from images originally labeled as 'Unknown'.

Rationale for Simplification

This reclassification was implemented to facilitate a more straightforward evaluation of how adversarial attacks could skew the model's damage assessments, impacting insurance claim processes. The new grouping allows for a clearer distinction between different levels of damage severity, essential for training robust machine learning models to accurately predict and categorize vehicle damage based on visual cues.