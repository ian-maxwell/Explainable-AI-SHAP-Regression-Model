Ian Maxwell
[Student ID Removed]
[Student Email Removed]

# Interpreting Fuel Efficiency: A SHAP-Based XAI Approach

This project focuses on **Explainable AI (XAI)** using **SHapley Additive exPlanations (SHAP)** to interpret an XGBoost regression model. Using the classic MPG (miles per gallon) dataset, the goal is to move beyond "black-box" predictions and understand _why_ a model predicts specific fuel efficiency values.

---

## 📊 Key Findings

By analyzing the SHAP values generated from the XGBoost model, we observed the following:

- **Primary Drivers:** **Vehicle Weight** is the most significant predictor. Higher weight consistently correlates with a strong negative SHAP value, significantly reducing predicted MPG.
- **Technological Progress:** The **Model Year** feature shows a clear positive trend—newer vehicles (higher year values) generally contribute positively to the MPG prediction, reflecting improvements in engine efficiency over time.
- **Non-Linearity:** Unlike simple linear regression, the SHAP scatter plots reveal non-linear relationships between **Horsepower** and efficiency, particularly in how it interacts with vehicle weight.

[Image of SHAP summary plot showing feature importance and impact]

---

## 🛠 Setup & Installation

### Prerequisites

- **Python 3.x**
- **VS Code** (Recommended)

### Required Libraries

Install the necessary stack via pip:
bash
pip install -r requirements.txt
or alternatively, you can install the core libraries manually:
pip install pandas numpy matplotlib seaborn xgboost shap ipython

---

## 🚀 Execution Options

1. **The first python file is for very quick simple execution it is called "Ian Maxwell 3190 SHAP project - quick.py".** You will have to open the project in vscode (recommended) and ensure you have installed python and the libraries listed above.

   - **Note:** The only issue here is that matploylib is not great at plotting the graphs using the SHAP library and thus some graphs will be cutoff. For perfect graphs you must do option 2.

2. **For perfect SHAP graphs feel free to do this option although it requires a couple of more steps. This second file is called "Ian Maxwell 3190 SHAP project - Jupyter extension.ipnyb".**
   You will have to open the project in vscode (recommended).

   - **Jupyter Extension:** Ensure you have the Jupyter extension installed in VS Code. If not: Open VS Code, go to the Extensions view (Ctrl+Shift+X or Cmd+Shift+X on macOS) search for “Jupyter” (by Microsoft), and install it.
   - **Notebook Kernel:** VS Code will prompt you to select a Python kernel (interpreter) when running the notebook.

You can execute the program by pressing the **play button** in the top left and scrolling down in the code to see the outputs.

---

## 📂 Critical Data Note

- **Note:** Both files require the `mpg.csv` file to be in the local directory so ensure the console/terminal reflects this. If the file is not in the same directory as your script or notebook, the program will not run.

---

_This project was developed as part of an exploration into Explainable AI techniques for the interpretability of Machine Learning models._
