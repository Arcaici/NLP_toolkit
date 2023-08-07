from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_frequency_classes(df, labels_column = "label"):
    # take in input a pandas Dataframe and a string that indicate the labels column name
    # and plot a frequency bar chart of the classes
    df[labels_column].value_counts(ascending= True).plot.barh()
    plt.show()

def plot_words_per_text(df, text_column= "text", labels_column = "label"):
    df["word_in_text"] = df[text_column].str.split().apply(len)
    df.boxplot("word_in_text", by=labels_column, grid=False, showfliers=False, color="blue")
    plt.suptitle("")
    plt.xlabel("")
    plt.show()


def plot_confusion_matrix(y_pred, y_true, labels_list):
    # Given a list of predictes labels and the list of the true labels
    # the function plot a confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize= "true")
    fig, ax = plt.subplots(figsize=(11,11))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = labels_list)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar= False)
    plt.title("Normalized confusion Matrix")
    plt.show()


def plot_models_score(scores_dict, score_type = "F1"):
    # Passing a dict with (key : value) pairs as (model_name : score),
    # function plot a hist comparing all models by score
    fig, ax = plt.subplots()
    df = pd.DataFrame.from_dict(scores_dict)
    df.plot(kind="bar", ylabel="Score", rot=0, ax=ax)
    ax.set_xticklabels([score_type])
    plt.legend()
    plt.show()

