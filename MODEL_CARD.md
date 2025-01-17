# Model Card

Final Metrics

* Accuracy of base model: [77.51%] — This represent the model's accuracy which has no attack or defense parameters in it
* Accuracy of base model with Attack 1: [16.87%] —  This represents the model's accuracy which has no defense parameters in it and is exposed to attack 1
* Accuracy of base model with Attack 2: [39.34%] — This represents the model's accuracy which has no defense parameters in it and is exposed to attack 2
* Accuracy of model with Defenseive Technique 1: [83.6%] — This represents the model's accuracy which has defense parameters in it and is exposed to attack 1
* Accuracy of model with Defensive Technique 2: [67.21%] — This represents the model's accuracy which has defense parameters in it and is exposed to attack 2

Domain-Specific Metrics

* Labels: The images are categorized into good (i.e. no damage), low, medium and critical damage types.

Training and Evaluation

* Training Data: Utilized a balanced Car damage dataset from Kaggle, implemented data augmentation strategies including random flips and rotations to simulate real-world variations.
* Training Procedure: The CNN model was trained using adversarial training techniques. Specifically, it incorporated the Fast Gradient Sign Method (FGSM) during training to generate adversarial examples on-the-fly, which were then used to train the model to recognize and correctly classify perturbed inputs.
* Evaluation: The model was tested on both a clean and adversarially perturbed test set to evaluate performance and robustness. Evaluations were performed under varying conditions to ensure reliability across different scenarios

Known Limitations

* Adversarial Robustness: While the model shows improved robustness against certain types of adversarial attacks (e.g., FGSM), it may not perform as well against other, more sophisticated attacks not encountered during training.
* Transferability: The defensive capabilities may not generalize well to new data that significantly differs from the data seen during training.
* Scalability: Performance and robustness metrics may degrade as the size and complexity of input data increase.