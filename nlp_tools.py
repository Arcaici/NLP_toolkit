import pandas as pd
from sklearn.metrics import log_loss


def save_model_pred(df, y_pred, file_name, eval_loss = False):
    # This model save the dataset to csv file
    # this function has been created for save validation dataset
    # with validation prediction extra column

    if eval_loss:
        df["loss"] = eval_loss

    df["prediction"] = y_pred
    df.to_csv(file_name, index= False)
    print("Dataframe saved to :", file_name)

def samples_loss(model, prob_predictions, y_test):
    loss_values = []
    for i in range(len(y_test)):
        sample_loss = log_loss([1 if j == y_test[i] else 0 for j in range(model.classes_.size)], prob_predictions[i])
        loss_values.append(sample_loss)

    return loss_values

def df_sample_losses(df):

    print("\n Sample with highest losses \n")
    df.sort_values("loss", ascending=False).head(10)
    print("\n\n ")

    print("\n Sample with lowest losses \n")
    df.sort_values("loss", ascending=True).head(10)
    print("\n\n ")
