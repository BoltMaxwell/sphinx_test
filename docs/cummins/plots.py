"""
Some required functions for plotting

Author: Shrenik Zinage

I need to still add more functions for plotting
"""

__all__ = ["parity_plot"]

# Importing required packages
import scipy.io   
from scipy.io import loadmat, savemat    
from scipy import stats as st
import statsmodels.api as sm                            
import matplotlib.pyplot as plt                 
import seaborn as sns                           
import numpy as np                              
import pandas as pd                             
import matplotlib.pyplot as plt  
import scipy.stats as stats

# Parity plot
def parity_plot(y_test, y_pred, y_ecm, dataset_index):

    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot(y_test, y_pred, 'o', markersize=2, color="blue", label = "Predicted")
    plt.plot(y_test, y_ecm, 'o', markersize=2, color="red", label = "ECM virtual sensor")
    plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'k--')
    plt.xlabel('Test data', fontsize =18)
    plt.ylabel('Predicted data', fontsize = 18)
    plt.title('Parity plot', fontsize = 18)
    plt.legend(loc = "upper right")
    plt.tight_layout()
    # plt.savefig(f'plot_{dataset_index + 1}_gpr_egr_parity.pdf', dpi=300)
    plt.show()









