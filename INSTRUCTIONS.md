# Practical AI - Fall 2024 Final Project

The final project for the semester will simulate use of the techniques we have covered in class so far in a professional setting.

For this project, you have two options:

1. **Option 1: Adversarial AI Tournament / AI Red Teaming:** For this option, you are required to select a partner, develop one offensive and one defensive technique each, and then compete your techniques against each other. Your offensive technique should be an inference attack, and your defensive technique should be an attempt to detect the presence of the inference attack. Your inference attack (offense) and inference attack detector (defense) must be original work (built using one of the toolkits we have discussed in class - you do not need to build adversarial attack techniques from scratch!). After you and your partner compete your techniques against each other, you must submit a report in the form of a Jupyter notebook detailing the results of your red-team exercise.

2. **Option 2: Individual Research:** The goal of this option is to provide students an opportunity to perform original research in open problems of interest. A *problem* is an unsolved issue which causes a pain in a being, organization, or society. For this option, you are required to select an open problem in modern machine learning (perhaps a dataset or task that is difficult to solve, or perhaps one of the topics in class we have discussed that still need more research to advance the state of the art.) You are not required to solve the unsolved problem that you select, but you are required to extract insights from the problem which are completely original.

In both projects, the end deliverable is a professional report detailing your red-teaming exercise or original research into a problem of interest, and detailing the processes you employed, following all of the best practices discussed in class. The report must provide original insights into the attacks and defenses you studied (option #1) or the original research you performed. When deciding what original insights you want to extract, ask yourself, would someone want to work with me on this problem in a professional setting if they saw the insights I extracted, or would they assume that anyone else (or any AI tool) could have easily extracted the same insights? The goal is to produce something only you could have produced.

No starter code will be provided. The goal of the project is to independently produce technical work and associated technical communication which extracts and professionally reports on insights gained from a data set about a problem of significance to a field you are interested in. Here we specify the project deliverables, requirements, grading criteria, and policies on collaboration with humans and AI tools.

Choosing a problem which pertains to your general field, research, academic, or professional interests is highly encouraged. If you are not sure what problem to pick, problem choices can be discussed in office hours.

Our class final exam period is scheduled for **Thursday, December 19**. Your project is due by 11:59 PM the following day, **Friday, 12/20**. You do not need to show up to class for this final. Use the time to continue working on your project if needed.

The final grades for our class are due by **Monday, December 23rd**. No work can be accepted after this date!

## Deliverables

Four project deliverables shall be provided.

* **Project Abstract**: Write one paragraph in your notebook specifying what problem you are going to be analyzing, where you can get data about that problem, what techniques you think you might be able to apply to it, and what insights you plan to attempt to extract from it. The project abstract is due the same day as the project itself, but if you are unsure of your topic and have not discussed it already, it is highly recommended to review your topic idea with your instructor first. Sending via email or discussing in office hours are both acceptable.
* **Project Notebook**: This is the main artifact of the final project. It should be a professionally written report on your problem and the insights you gained from it, including any techniques from class you used to extract those insights and visualizations which display them.
* **Project Modules**: Rather than developing large chunks of code in your Jupyter notebook, you are required to implement complex functions in their own modules and call these modules from your notebook. There should be a minimal amount of code in your Jupyter notebook.
* **README file and ML documentation**: You are required to produce a `README.md` file. The sections that must be in your README are detailed below. You are also required to put basic information about your dataset in `DATA_CARD.md` and basic information about your model in `MODEL_CARD.md`.

All deliverables shall be provided by **11:59 PM** at the end of the exam period. Please submit via git but note that no points will be deducted for an improper git submission for this project. Git submission mechanics were assessed in the beginning of the semester. The goal of this assignment is to assess your ability to extract original insights to tell a data driven story that is entirely your own.

## Requirements

The following project requirements on format, technical depth, insight, software, technical communication, and use of techniques from class shall be adhered to.

### README File

At minimum your README file must include the following information.

1. **Purpose:** This section includes a detailed description of the purpose of your project that a user who does not have familiarity with it already will understand.
2. **Usage Instructions:** This section includes subsections with instructions on how to clone your project, install your project, and use your project, including sample code snippets in Markdown where applicable.
3. **Known Issues:** This section describes any known bugs or issues.
4. **Feature Roadmap:** This section describes the next features that will be added to the project.
5. **Contributing:** This section describes how interested parties can contribute to the project.
6. **License:** This section states which license the project is distributed under and links to the license file. If parts of the project are copyrighted or proprietary then that is described here.
7. **Contact:** This section lists an email for the project's main point of contact.

An example README file can be found [here](https://github.com/othneildrew/Best-README-Template). You do not need to include all information in this README, but it provides a great starting point to meet the requirements above.

### MODEL_CARD.md and DATA_CARD.md

Use `MODEL_CARD.md` to report basic information about your final model:
* Final metrics (which ever are appropriate)
* Any final domain specific metrics
* Brief notes on how the model was trained and evaluated
* Any known limitations

Use `DATA_CARD.md` to report basic information about your dataset:
* Where the data came from
* What the purpose of the data set is
* Who authored the dataset
* Who own the dataset
* What the license of the dataset is
* Any significant processing that was performed on the data

Both these forms of documentation will contain much more information in a professional setting, but this is sufficient for a class project.

### Project Notebook Outline

Your project notebook may contain as many sections as you need to address your problem. At minimum, recommended sections include:

* **Title**: Give your project a descriptive title.
* **Abstract**: Briefly describe your problem addressed, methods used, and results obtained in a single paragraph.
* **Introduction**: Introduce your project. What are you trying to achieve? What methods do you intend to use?
* **Problem Addressed**: Describe the problem you are addressing. Remember that a problem is something that causes a pain in a being, group, or society.
* **Motivation**: Motivate your work. Describe why the work is important and who will benefit / what the payoff will be if it is successful.
* **Previous Work**: Perform a literature review and describe any prior work that is related to yours.
* **Project Schedule and Budget**: How much will your project cost? What will the team look like to productionize your work? What milestones will indicate success? Use the project planning best practices from class to address this section.
* **Technical Approach**: Describe the technical approach you took and why it is justified.
* **Main Results**: Present your main results.
* **Future Work**: Describe any next steps for your project. What features are on your roadmap? Is more research required? In what aspects of the project?
* **Discussion**: Summarize your findings and discuss their implications.

There are no minium or maximum requirements on the length of each section. Use the amount of text required to address each section professionally. In general, most sections should be addressable in 1-2 short paragraphs, but this is not a hard limit or requirement.

You may choose to deviate from these section headings and this overall format, just be sure to adhere to the technical requirements below and demonstrate that they were adhered to in your report to receive full credit.

### Project Repository Structure

Your project repository shall have the following structure.

* `.devcontainer` - This is your development container. It will be defined for you in this class but you are welcome to modify it.
* `projectname` - This is the name of your Python package. It will be defined for you in this class but you are welcome to modify it. This is where the code for your project goes.
* `notebooks` - This where your notebooks go. Your notebooks should import packages from `projectname` rather than including large blocks of code.
* `tests` - This is where unit tests go. Unit tests are not required for this project, but are always required in production code. Often in this class, the tests are already written as part of a quiz.
* `.gitignore` - This file tells Git which files to ignore. It will be defined for you but you are welcome to modify it.
* `LICENSE` - The license file for the project. Change this file if desired.
* `poetry.lock` - This file defines the working configuration of Python packages that Poetry solved for. You should commit it with your repository.
* `pyproject.toml` - This file is generated by Poetry when a new project is started. It defines the packages and versions thereof that the project uses. It is written for you in this class but you are welcome to modify it.
* `README.md` - This is your README file.

Most of this is done for you in the project repository you will get from GitHub classroom but you must change the files as needed to meet the project requirements.

Inside the `projectname` folder, your should use standard Python conventions for director structure. For example:

```text
packagename/
    subpackagename/
        __init__.py
        subpackagemodule.py
    __init__.py
    module.py
```

### Guidance on Novel Insight

There are novel problems and old problems. There are also novel methods of attacking problems and old methods of attacking them. This means there are four types of contributions. Three of them are novel.

1. We can attack an old problem with an old method. This does not constitute a novel contribution, but can be valuable if we explain the solution in a way that is better than others who have applied these methods have done before.
2. We can attack an old problem with a new method. This is a novel contribution. We can show why our contribution matters by explaining why we think our new methods will bring new insights into the old problem that many others have studied before.
3. We can attack a new problem with an old method. This is also a novel contribution. Sometimes an old method can solve new problems that have emerged since the method was originally developed. These discoveries are quite exciting since they show us how existing tools can solve emergent problems.
4. We can also attack new problems with new methods. This is where cutting edge research is performed. There is a wealth of novelty to be found in this region of the problem-solution space!

Project notebooks shall make at least one novel insight. Any type 2, 3, or 4 insight is a novel one.

In a professional setting, a novel contribution can be pragmatically thought of as one which someone would hire you on the job to perform instead of hiring someone else or using an AI tool to obtain. **This means a novel insight is one that only *you* could have produced.** Novel insights come from breaking down unsolved problems into subparts and synthesizing your own unique personal experience to solve them. In an informal setting, a novel contribution is one that someone else equally knowledgeable about your field would be interested in discussing with you (and perhaps might even find interesting enough to pay for your meal or drink)!

### Technical Depth

Deliverables shall be of graduate level technical depth. Indicators of graduate level technical depth include the following.

* Multiple sources shall be surveyed and cited. If others have worked on your problem before, be sure to cite a few examples. If you use frameworks or techniques developed by others, be sure to cite your sources.
* The source of any data used shall be cited.
* Benefits, challenges, tradeoffs, and limitations of methods studied shall be discussed.
* The analyses performed shall show thorough consideration for the best practices discussed in class (see below for details).
* You do not need to solve your unsolved problem (the assignment deadline is far too short for that!) but all notebooks shall present novel ways of thinking about the topics selected using the techniques we have discussed in class.

### Software

The software used shall follow the principals of software hygiene discussed in class. All lengthy functions (> 20 lines) shall be written in reusable modules and imported into the notebook deliverable. The software written shall run outside your local environment with no trouble. This means functions must be written in a reusable manner which does not reference local paths.

Any Jupyter notebooks shall be saved in a run state so the results are readily visible. Seeing the results of your project must **not** require downloading large datasets or running your data cleaning, model training, and other computationally expensive steps. Some students are using datasets that are 10s of GBs and training fairly deep neural networks. It is not possible to grade close to 30 student projects by rerunning these steps. This means your grade will be based entirely on the results produced in your project deliverables and the quality of your code!

### Technical Communication

Project notebooks shall showcase professional technical communication skills. Project notebooks shall be divided into well structured sections and shall use markdown to specify headings for each section. Project notebooks shall use LaTeX within markdown to typeset equations (AI tools are great for helping with this). Project notebooks shall be written in full sentences and well-constructed paragraphs (3-10 sentences per paragraph). Writing shall guide readers through the material by narrating where they are in the document and explaining where they are going next.

### Use of Techniques from Class

Project notebooks shall use at least 5 tools we have learned about in class so far.

Tools we have studied include:

* PyTorch for deep learning
* Captum for explainability
* Scikit-Learn for machine learning
* Scipy for optimization
* Pandas for data management
* ONNX for model saving
* Adversarial Robustness Toolkit for adversarial testing / red teaming
* Seaborn, Matplotlib, etc. for data visualization
* Adversarial Robustness Toolkit or Cleverhans for robustness testing

Project notebooks shall use at least 5 techniques we have learned about in class so far. Techniques and best practices we have learned about are listed in the grading rubric below.

## Grading Rubric

### Originality (25%)

The following criteria will be used to assess the originality of a project.

#### Choice of Problem and Dataset

Project notebooks shall represent an analysis of a problem of interest to the student's field of choice and explain why it is of interest to this community. Notebooks lacking explanation of why a problem is of interest will lose **5 points**.

#### Original Steps to Solve the Problem

Project notebooks shall go beyond simply calling available functions from one or two popular libraries and shall instead take original steps to analyze and solve the problem by combining tools from many libraries in unique ways. Notebooks which use < 2 tools to perform a straightforward analysis (e.g., simply training some default sklearn models with no further experimentation) will lose **5 points**.

#### Original Insights Gained

Project notebooks shall extract at least one novel insight from the problem chosen. You do not need to solve the problem completely on such a short timeline, but you must try original ideas and extract original insights that no one else has extracted. Projects which do not extract a novel insight will lose **5 points**.

See the section providing [guidance on novelty](#guidance-on-novelty-and-insight).

#### Actionability of the Insights

The insights presented by project notebooks shall be actionable. For each insight presented, ask yourself, can the reader take action from it? What action would they take? If these questions are difficult to answer, then the information presented is not actionable. Three actionable insights shall be presented per project. A notebook without enough actionable insights will lose **5 points**.

#### Original Exposition of the Insights

The insights shall be exposited in an original manner. Each insight should be explained in an original manner, in your own words. Use analogies and other explanatory devices to help the reader see the insights you have extracted from the data and take action on them.

#### Original Code and Work

All code and writing must be the student's own work. See the policies on [collaboration](#policy-on-collaboration), [use of AI](#policy-on-use-of-ai), and the course policy on academic integrity. Any violations of these policies will result in points deduction or a zero-grade, depending on the violation and severity.

### Software Hygiene (25%)

#### Use of Modules

Any functions longer than 20 lines of code shall be implemented in importable modules. If any functions greater than 20 lines of code implemented within project notebooks, **5 points** will be deducted.

#### Ensure Your Code Runs in Any Environment

Project code shall be executable outside your personal development environment. Project code shall not contain local paths specific to your machine. Projects which include code that will not be executable outside your local environment will lose **5 points**.

#### High Quality Documentation

All project deliverables shall provide high quality documentation. All functions shall have docstrings. Use comments, but do not comment every line of code. Write code that is self-explanatory and does not need a comment for every line. Ensure the notebook reads like a professional report. Notebooks and modules without high quality documentation will lose **5 points**.

#### Use of a Style Guide

Project notebooks and code shall be styled with `black`. You do not need to use automatic formatting (though you are encouraged to). Running `black` once before submission is sufficient. Projects clearly unstyled will lose **2.5 points**.

#### Quality of Commit Messages

Write high quality commit messages using the guidance learned in class. A good commit message should read `"When applied, this commit will...[insert your message here]"`. Poor quality commit messages will lose **2.5 points**.

#### Descriptive Variable and Function Names

Variable names shall be descriptive nouns. Function names shall be descriptive verbs. One letter variable names shall not be used. Variables which store measured quantities shall include the units of the measured quantities in their names. Variable names referencing mathematical symbols shall spell out the name of the symbol and still be descriptive, e.g., `gamma_euler_mascheroni_constant` not `gamma` and not `g`. Poor quality variable names will lose **2.5 points**.

#### Dead Code

Project notebooks shall contain no commented out or dead code. Notebooks with dead code with lose **2.5 points**.

### ML Best Practices (35%)

You are required to use your best judgement to apply ML best practices from class as appropriate for your problem. At minimum, you must:

* Follow all "basic basic" practices and practices marked mandatory (**10 points**).
* Include a Project Schedule and Budget discussion in your report addressing the ML project planning considerations we have discussed in class in 1 or 2 paragraphs (**10 points**).
* Discuss how at least **one** of the items **from each of the applicable sections** (i.e., include one item per section in your discussion) below was applied in your project (**15 points**). You only need a few sentences per item.

Items marked "Mandatory" are required in this project and minimum standards for a professional ML project. If you employ lessons learned in class, following the mandatory items will come naturally. Items marked "Best Practice" are highly recommended best practices that should be applied where applicable but require judgement in application to ensure the correct best practice is applied in the correct context. Items marked "Technique" are techniques from our playbook to be applied when a challenge which motivates them occurs.

#### Basic Basics
* (Mandatory) Projects shall not train on the test data. This should go without stating.
* (Mandatory) Account for data drift. Be sure to describe how you will be able throw away drifted samples or retrain your model to account for drift.
* (Mandatory) Do not pick a problem outside your model's learning capacity given your available computing resources.
* (Mandatory) Use pre and post guardrails for input and output validation.
* (Mandatory) Apply Occam's razor to focus on the most narrow task.

#### Basic Basic Tuning
* (Mandatory) Be sure to visualize your data to confirm the labels are accurate and edit inaccurate labels.
* (Mandatory) Be sure to throw out bad or suspect data.
* (Mandatory) Be sure to define your loss function and "definition of good" using domain knowledge.
* (Mandatory) Define your learning paradigm (supervised, unsupervised, reinforcement learning, hybrid approaches) early.
* (Mandatory) Tune learning rate as a first step in your hyperparameter tuning process.

#### Project Planning
* (Mandatory) Choose a project planning paradigm (V-model, Russel and Norvig, Goodfellow, Google) and describe the phases of your project using that terminology.
* (Mandatory) Discuss if ML is the best solution and what non-ML baselines have been implemented before.
* (Mandatory) Define a definition of good performance and a performance goal before starting.
* (Mandatory) Specify which problem you're solving for your users and which parts you are solving with machine learning.
* (Mandatory) Assess the ethics of your use case, are there any ethical or legal issues to consider? (autonomous weapons, privacy, security, fairness, trust, safety, potential harm to users).
* (Mandatory) Discuss whether you will need to make any coverage-accuracy tradeoffs.
* (Mandatory) Discuss if it will be feasible to collect data, how much data will be required?
* (Mandatory) What are the expected requirements for queries per second?
* (Mandatory) Discuss how much budget and how much staff you would need to productionize this project - remember that ML projects are nonlinear.
* (Mandatory) Does your project show enough promise for early productionization? Discuss what results you wish to achieve before starting to apply staff to build the production version of your project.
* (Mandatory) What is your plan to get user or stakeholder feedback? What kind of feedback is reasonable to obtain?

#### Architecture
* (Mandatory) Separate training and inference code.
* (Mandatory) Serialize configuration files using yaml or json.
* (Mandatory) Serialize model files using ONNX.
* (Mandatory) Define your model as a class which extends one of PyTorch's classes.
* (Mandatory) If using a custom dataset, define your dataset as a class which extends PyTorch's Dataset class.

#### Data Pipeline
* (Mandatory) Remove exceptional cases from data.
* (Best Practice) Engineer features where appropriate.
* (Best Practice) Use unsupervised learning as a first step to help label data if data is unlabeled or labels are suspect.
* (Mandatory) Remember to annotate potentially ambiguous labels so those cases can be analyzed specially.
* (Mandatory) Note if classes are unbalanced and under / over sample to balance classes as required.
* (Best Practice) Perform exploratory data analysis to learn more about the data - look for correlations, consider visualizing data in a low dimensional space using PCA, T-SNE, or auto-encoding.
* (Technique) Consider discretizing features to remove noise.
* (Technique) Consider applying noise removal techniques like independent component analysis (ICA).

#### Sensing
If gathering data from a sensor:
* (Mandatory) Quantify performance with respect to SNR. Show SNR in decibels (dB).
* (Mandatory) Characterize mean accuracy with respect to SNR over repeated experiments.
* (Mandatory) Quantify performance with respect to range.
* (Best Practice) Use correlations to detect patterns in noise as appropriate.
* (Best Practice) If applicable, consider using a model of the way your sensor collects data to determine if the data collection process will impact model performance.
* (Mandatory) Ensure you discuss the type of sensor collecting data and how the sensor works.
* (Mandatory) Ensure you discuss sources of noise in your sensor.

#### Training Models
* (Mandatory) Evaluate multiple architectures.
* (Mandatory) Attempt multiple experiments - **ensure all experiments are replicable by documenting parameters employed**.
* (Mandatory) Use seeds for random number generators. Define configuration parameters in files as appropriate.
* (Best Practice) Employ transforms to augment your data as appropriate.
* (Mandatory when using PyTorch) Employ a data loader to get data from your disk to batches in RAM that PyTorch can use to train your models.
* (Mandatory) When tuning hyperparameters, change single parameters in a small way and record progress.
* (Mandatory) Consider the no free lunch theorem. Be sure to discuss this in your report.
* (Best Practice) Consider designing features to increase separability (equations, quantizations, encodings).
* (Mandatory) Tune hyperparameters to increase or decrease model capacity as required.
* (Technique) Consider hybrid architectures of deep learned and statistical techniques as appropriate.

#### Explainability
* (Mandatory) Consider performance vs. explainability tradeoffs and discuss them in your report.
* (Mandatory) Consider the type of explainability best suited for your problem and discuss it in your report. Is strict interpretability (by the Russel and Norvig definition) required for your problem?
* (Mandatory) Consider the multiplicity of good models problem - if there are many solutions that fit your data, which is the most robust, explainable and reliable? Discuss this in your report as appropriate.
* (Technique) Consider using surrogate models as appropriate.

#### Robustness and Hardening
Apply the following robustness and hardening techniques as appropriate.
* (Technique) Augment data to increase robustness to data transformations.
* (Technique) Select model architectures deliberately using domain knowledge and/or comparative testing as appropriate.
* (Technique) Exploit expert knowledge of the problem to eliminate noisy or unreliable features, and/or engineer features robust to noise.
* (Technique) Employ filters (e.g., simple averaging or more advanced techniques as appropriate).
* (Technique) If noise is below a certain level, consider employing nonlinearity as a defense by removing noise below a threshold.
* (Technique) Detect drifted or out of distribution samples and remove them.
* (Technique) Regularize your model, perhaps with a custom loss function which rewards characteristics deemed robust by experts.
* (Technique) Sanitize your data by removing low quality, anomalous, or potentially malicious inputs.
* (Technique) Consider training on adversarial examples or model distillation in complement or in place of regularization.

#### Red Teaming
If selecting project option #1, consider applying the following.
* (Best Practice) Test unnatural or unusual inputs to determine if model behavior is still expected.
* (Mandatory) Consider if confidentiality, integrity, or availability can be compromised as appropriate for the attack surfaces available.
* (Technique) Consider whitebox, greybox, and blackbox attacks as applicable.
* (Technique) Consider poison data, membership inference, availability, or model stealing attacks.
* (Technique) Consider exploiting known weak defenses, such as gradient masking or label noise.
* (Technique) Consider employing model stealing as an opening move.

#### Test and Evaluation
* (Mandatory) Run explainability to evaluate candidate models. Chose the appropriate explainability technique for your model and problem from the many techniques discussed in class!
* (Mandatory) Run a conformance test after final models are cut.
* (Best Practice) If appropriate, robustness test your model against degree of challenging transforms (such as blur, noise, darkness, etc.).
* (Mandatory) Extract basic training metrics including loss, confusion matrix, false positives and misses as appropriate.
* (Mandatory) Compute final metrics in the problem domain that the stakeholder (not the ML engineer) cares about, e.g., do not show confusion matrices for a buy/sell indicator prediction problem without assessing the returns generated in a back-test of trades executed based on the algorithm's prediction.
* (Mandatory) Test and compare performance against a baseline to show if your technique improves against the baseline.

You are encouraged to take this checklist with you for ML projects you take on in the future, and add to it as the field evolves and you gather knowledge in your specific domain of expertise!

### Basics (15%)

#### Proper Use of Functions and Lambdas

Functions and lambdas shall be implemented correctly without logical or execution errors (3 points).

#### Proper Use of Conditional Statements, For Loops

Conditional statements and for loops shall be implemented correctly without logical or execution errors (3 points).

#### Proper Use of Data Structures

The correct Python data structures shall be used (Lists, Dictionaries, Tuples, etc.) for the correct applications (3 points).

#### Proper Use of NumPy

NumPy shall be used for code that operates with arrays. Array operations shall be used instead of for loops (3 points).

#### Proper Use of DataFrames

DataFrames shall be used properly (3 points).

* The `apply()` function shall be used to iterate a function over a DataFrame.
* Use `to_list` and `to_numpy` functions shall be used to extract lists and arrays from a DataFrame.
* Rows, columns, and elements shall be accessed properly.
* Masks shall be used correctly to select data.

## Policy on Collaboration

Students are allowed to collaborate on the final project (e.g., to debug an error), however, all work in the deliverables must be your own and the following rules shall be adhered to.

* Do not use the same code as another student.
* Do not extract the same insights as another student.
* Do not collaborate with others outside the class (e.g., a student who took the course previously).
* Do not submit work that is not your own.

Any violations of these policies will result in a grade of **zero** for the final project. In a professional setting, it is important to extract insights from data that our competition cannot. Why hire a new developer or a new firm if they produce the same insights a current contract is producing? Originality is critical to maintaining a competitive edge and will be graded accordingly.

## Policy on use of Foundation Models

Using foundation models to help you write code is allowed. However, do not do violate any of the following rules.

* Do not provide code written by AI tools without adding any original work to it. If you just learn to use AI to perform tasks, why hire you when the AI can do the work cheaper and without getting tired?
* Do not provide code written by AI without checking that it actually does what you think it does.
* Do not use AI in a way that violates the Veterinary Dentists law, i.e., ensure you have knowledge of the AI tools you are using AND knowledge of the domain in which you are applying them. For example, do not use AI to perform analysis of a dataset in a field you know nothing about without researching that field extensively to ensure the AI results are correct, and do not use AI to perform an analysis in a field you are an expert about unless you provide citations to the AI tools you are using and can demonstrate you understand how they work, and how they arrived at the insights you acquired from them.

Using AI tools to help write code is allowed, but no two students should have the same code. The first occurrence of code that is exactly the same (or the same with trivial modifications) as another student will lose **5 points**. The second occurrence will lose **10 points**. The third will lose **20 points**, and so on. In a professional setting, it is important to extract insights from data that AI cannot alone. Why hire a new developer or a new firm if they produce the same insights that the hiring body can produce with AI? AI can perform many tasks cheaper and faster than humans. Use this assignment to practice using AI as a tool to expand your own original insights rather than as a tool to replace yourself!

## Anti-Patterns to Avoid

The following anti-patterns shall be avoided. Ask if you have a question about any of these. Unless specified elsewhere, each occurrence of these anti-patterns will result in a **2.5 point deduction** so please be careful!

### AI Anti-patterns

* Do not start a project without defining the problem to be addressed.
* Do not start an ML project without assessing if ML is the best solution.
* Do not start a new ML project without assessing a baseline to improve against.
* Do not train a complicated model without evaluating a simple model.
* Do not assume a waterfall development schedule will work for MLOps projects.
* Do not assume human labels are infallible or unbiased.
* Do not employ ML if there is no mechanism to detect data drift and retrain to account for data drift as needed.
* Do not couple training and inference code.
* Do not use insecure formats for final model serialization.
* Do not use unclean data.
* Do not use data that has not been checked for potentially ambiguous or inseparable cases.
* Do not use data from a sensor without understanding how the sensor works.
* Do not use data from a sensor without quantifying the performance limits of the sensor.
* Do not use a single architecture without performing an architecture search.
* Do not ignore the no free lunch theorem - make the model's task as narrow as possible.
* Do not tune multiple hyperparameters at once.
* Do not cut models for production without running explainability.
* Do not wait until final training runs to check explainability results.
* Do not show performance curves without repeating experiments to ensure results are repeatable.
* Do not push models to production without a conformance test.
* Do not show performance results to stakeholders without domain-specific system-level performance results.

### Visualization Anti-Patterns

* Do not show plots with no title.
* Do not show plots with no axis labels.
* Do not show plots where axis ticks and associated text run together.
* Do not show plots where the axes labels or titles are too small to see.
* Do not show plots with more labels or more x-ticks than anyone can reasonably see.
* Do not show plots which show multiple relationships with no legend.
* Do not show subplots that run into other plots.
* Do not show dates in non-date format (e.g., year 2000.5).
* Do not use grid lines unnecessarily.
* Do not show noisy data without extracting a smooth trend or determining that no trend exists.
* Do not show axes in unnatural units (please scale data so that there are only 3 numbers before the decimal place, e.g., 50 km rather than 5e4 m).
* Do not showing statistical data without showing uncertainty.
* Do not use one-dimensional scatter plots to convey a distribution over a single independent variable, rather than using histograms, violin plots or other appropriate means of showing statistical data.
* Do not plot too many subplots without expanding the width or height of the figure to accommodate them.
* Do not show a correlation matrix without investigating the correlations for realism and investigating variables that are not correlated for mutual information with other variables (e.g., do not simply use `sns.PairGrid` or `sns.heatmap` without doing any other work)!

### Typesetting Anti-Patterns
* Do not write mathematical symbols in informal notation, e.g., `e^(sin(x))` rather than $e^{\sin(x)}$.
* Do not include cells that produce errors in your finished notebook.
* Do not leave necessary warnings unsuppressed in your notebook.
* Do not leave unnatural notation from the code in the plot labels (e.g., do not show a Y axis labelled `my_dataframe_col_with_altitude` instead of taking the time to label the axis `Altitude (km)`).
* Do not reference a quantity without specifying its units unless working with purely mathematical objects.
* Use proper capitalization in plot labels and axes.

### Technical Communication Anti-Patterns
* Do not turn in notebooks with spelling mistakes (please use a spell checker).
* Do not turn in notebooks with incomplete sentences.
* Do not use an exploratory tone, e.g., do not write "here we try X, it didn't work, so instead let's try Y". Exploratory tone is encouraged in exploratory notes, but this project will produce a final deliverable!
* Do not forget to make the goal of each notebook section clear to the reader.

### Software Anti-Patterns
* Do not develop large blocks of code inside a notebook. Instead, make them available as reusable modules.
* Do not use paths specific to your local machine or personal development environment.
* Do not use for loops where NumPy can be used instead.
* Do not write overly lengthy lines (> 120 characters, or > 80 if you use `black`).
* Do not comment every line.
* Do not use single letter variable names.
* Do not leave commented out code in your notebook.

### Anti-Patterns Showing Lack of Originality
* Do not use the same exact code as another student (this **will result in a zero** - see collaboration policy and policy on academic integrity).
* Do not use code directly from an AI tool without performing any original work on the code (this will be penalized - see requirements on originality).

## Guidance on Using Open Source Tools

For any non-standard tools, be sure to investigate the following aspects of the tool before adopting it.

* Who is it made by? Are they a reliable source? What are their credentials?
* How many people was it made by? A team or an individual?
* Who is it maintained by? How many people?
* How often is it updated? When was it last updated?
* How well documented is it? Is there a support community?
* Is there contributor documentation? If you need a feature is there a clear way to request it?
* What are the dependencies? Will they conflict with any other project dependencies? How standard are they and are they well maintained?
* Have there been independent evaluations?
* Is there backing in the methods used in peer reviewed literature?
* What is the license of the project? Is it free to use for your use case? Are there restrictions that prevent your use case?
* When working professionally:
    * If there is a license, can your project's budget incur the cost?
    * Will the license affect your profit margins?
    * Does your organization have an open source approval process?
    * How quickly will the inclusion of this dependency by approved? Will approval change project schedule?
    * Does it need to be installed on your production system or just on your development system? Are there additional approvals required to install it on your production system?
    * Who from your organization needs to be consulted / informed? Your software architect? Your manager?
    * Does your customer need to be informed? Do they have expectations on the dependencies your project will have?