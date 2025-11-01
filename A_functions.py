#Here we will keep our functions to use in the jupyter notebooks so that we can easily share functions.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def quantile_objective():

    def objective_function(preds, dtrain):

        #print("I happened")

        labels = dtrain.get_label()
        errors = preds - labels # Note: Using preds - labels aligns with I(preds > y) convention

        # Calculate Gradient
        grad = np.where(errors > 0, 1 - 0.2, -0.2)

        # Calculate Hessian (using a small constant approximation)
        # A small positive constant helps ensure stability in the algorithm.
        # The exact value might require tuning or experimentation.
        hess = np.full_like(preds, 1.0) # Or a smaller value like 1e-6

        return grad, hess
    return objective_function

def get_prior100_weights(training_set, rm_id): #Training set should be either train_y or final_df
    data = training_set[training_set["rm_id"] == rm_id]
    weight = data["daily_weight"]
    return(weight.iloc[-100:])




def plot_id_simple(id, df):
    #Filter for specific ID
    df = df[df["rm_id"] == id]

    x = df["date"]
    y = df["cumulative_weight"]

    sns.set_theme()
    plt.plot(x,y)
    plt.title(label=f"Cumulative weight for ID: {id} in time period {x.min()} to {x.max()}")
    plt.xlabel("Dates")
    plt.ylabel("Cumulative weight")

print("imported functions from local folder")