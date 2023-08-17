import pandas as pd
from datasets import Dataset
from sklearn.metrics import log_loss
import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def remove_special_chars(df, text_column= 'text'):
    # this function takes in input a dataframe and a string containing the text column name
    # and return the dataframe with the text column cleaned from special chars
    df[text_column] = df[text_column].apply(lambda text: re.sub(r"[^a-zA-Z\s]", "", text))
    return df


def remove_stopwords_and_punkt(df, text_column="text", legal_stopwords=False):
    # this function takes in input a dataframe, a string containing the text column name and a list of legal words.
    # the function return a dataframe with the text column cleaned from stopwords and punctualization
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))

    if not legal_stopwords == False:
        stop_words = stop_words.union(legal_stopwords)

    def remove_stop_words(text):
        words = word_tokenize(text.lower())
        clean_words = [word for word in words if word not in stop_words]
        return " ".join(clean_words)

    df[text_column] = df[text_column].apply(remove_stop_words)

    return df


def remove_short_text(df, text_column="text", min_length=3):
    # this function takes in input a dataframe, a string containing the text column name and the minimum length
    # of words and return a dataframe with the text column with words more long then min_length param

    nltk.download('wordnet')
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()

    def remove_words(text, min_length=3):
        words = word_tokenize(text)
        words_filtered = [word for word in words if len(word) >= min_length]
        return " ".join(words_filtered)

    df[text_column] = df[text_column].apply(lambda text: remove_words(text, min_length))

    return df


def lemmatize_text(df, text_column="text"):
    # this function takes in input a dataframe, a string containing the text column name and
    # return a dataframe with the text column lemmatized

    nltk.download('wordnet')
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()

    def lemmatize(text):
        words = word_tokenize(text.lower())
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized_words)

    df[text_column] = df[text_column].apply(lemmatize)

    return df

def chi2_feature_selection(X_train, X_test, y_train, k):
    # this funtion takes in input train samples, test samples, train labels and k (number of features to select)
    # and return the train and test features selected using the chi-square correlation metrics
    chi2_features = SelectKBest(chi2, k = k)
    X_train = chi2_features.fit_transform(X_train, y_train)
    X_test = chi2_features.transform(X_test)
    return X_train, X_test

def save_model_pred(df, y_pred, file_name, eval_loss=False):
    # This model save the dataset to csv file
    # this function has been created for save validation dataset
    # with validation prediction extra column
    if (str(type(df)) == "<class 'datasets.arrow_dataset.Dataset'>"):
        df = Dataset.to_pandas(df)

    # if samples loss present is added to df
    if not (eval_loss == False):
        df["loss"] = eval_loss
        print("enter")

    #adding predictions
    df["prediction"] = y_pred
    df.to_csv(file_name, index=False)
    print("Dataframe saved to :", file_name)
    return df


def samples_loss(model, prob_predictions, y_test):
    #This function calculate loss value for each sample
    loss_values = []
    for i in range(len(y_test)):
        sample_loss = log_loss([1 if j == y_test[i] else 0 for j in range(model.classes_.size)], prob_predictions[i])
        loss_values.append(sample_loss)
    return loss_values


def df_sample_losses(df):
    # This function print a table with the 10 samples with the highest loss
    # and the lowest loss
    df = df[["text", "loss", "prediction", "label"]]
    print("\n Sample with highest losses \n")
    print(df.sort_values("loss", ascending=False).head(10))
    print("\n\n ")

    print("\n Sample with lowest losses \n")
    print(df.sort_values("loss", ascending=True).head(10))
    print("\n\n ")
