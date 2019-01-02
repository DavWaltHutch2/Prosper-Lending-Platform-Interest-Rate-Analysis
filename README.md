# Prosper Lending Platform Interest Rate Analysis

This applications leverages the power of analytics, to help borrowers predict interest rates on the Prosper Lending Platform based on their financial health and financial needs.  For more details reference the document report.pdf.

### Prerequisites

The application is built on top of serveral python modules.  To install all relevant modules, use the requirments.txt file that has been provided.    
```
pip install -r requirments.txt
```

### Run Application 
To perform exploratory analysis on the Prosper dataset, use the command below.  All charts will be saved to the folder labeled "charts".   
```
python prosper_exploratory_analysis.py
```

To analyze different models (Random Forest, ADA Boosting, Gradient Boosting, K Nearest Neighbors) on the Prosper dataset, use the command below.  All charts will be saved to the folder labeled "charts".  
```
python prosper_model_analysis_v6.py
```

To serialize the model used in the application, use the command below.  The model will be saved to the folder labeled "model".
```
python pickleModel.py
```

To run the application, use the command below.
```
python project.py
```





