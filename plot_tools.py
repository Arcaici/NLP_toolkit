from datasets import Dataset
from matplotlib import pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_frequency_classes(df, labels_column="label"):
    # take in input a pandas Dataframe and a string that indicate the labels column name
    # and plot a frequency bar chart of the classes
    if (str(type(df)) == "<class 'datasets.arrow_dataset.Dataset'>"):
        df = Dataset.to_pandas(df)

    df[labels_column].value_counts(ascending=True).plot.barh()
    plt.show()

def plot_words_per_text(df, text_column= "text", labels_column = "label"):
    # take in input a pandas Dataframe, a string that indicate the labels column name
    # and a string that indicate the text column name and return a plot showing the
    # frequencies of word in text feature

    if (str(type(df)) == "<class 'datasets.arrow_dataset.Dataset'>"):
        df = Dataset.to_pandas(df)

    df["word_in_text"] = df[text_column].str.split().apply(len)
    df.boxplot("word_in_text", by=labels_column, grid=False, showfliers=False, color="blue")
    plt.suptitle("")
    plt.xlabel("")
    plt.show()

def worldcloud(df, text_column = "text"):
    # take in input a pandas Dataframe and a string that indicate
    # the text column name and return a wordcloud plot
    text_data = df[text_column].dropna().str.cat(sep=' ')
    # Assuming your text data is stored in the variable 'text'
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

def twenty_most_common_words(df, text_column = "text"):
    # take in input a pandas Dataframe and a string that indicate
    # the text column name and return a histogram with the twenty
    # most common words and them occurrences
    df[text_column] = df[text_column].apply(lambda x: str(x))
    all_reviews = ' '.join(df[text_column])
    all_words = all_reviews.split()
    word_counts = Counter(all_words)
    word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index', columns=['count'])
    word_counts_df = word_counts_df.reset_index().rename(columns={'index': 'word'})
    # Ordina le parole per conteggio in ordine decrescente
    word_counts_df = word_counts_df.sort_values(by='count', ascending=False)
    # Crea un count plot delle prime N parole più comuni
    top_n = 20
    plt.figure(figsize=(12, 6))
    sns.barplot(x='count', y='word', data=word_counts_df.head(top_n))
    plt.xlabel('Count')
    plt.ylabel('Word')
    plt.title(f'Top {top_n} Most Common Words')
    plt.show()


def word_distribution_over_class(df, word, text_column="text", class_column='class_column'):
    # take in input a pandas Dataframe, a string that indicate
    # the text column name, a word, and return a histogram with
    # the distribution of the word over the class

    def sum_word_occurrences_by_class(df, word, text_column='text_column', class_column='class_column'):
        occurrences_by_class = []

        for class_label, group in df.groupby(class_column):
            total_occurrences = 0
            for text in group[text_column]:
                total_occurrences += text.lower().split().count(word.lower())
            occurrences_by_class.append({'class': class_label, 'occurrence': total_occurrences})

        return pd.DataFrame(occurrences_by_class)

    word_occurrences_per_class = sum_word_occurrences_by_class(df, word, text_column, class_column)

    sns.barplot(y='class', x='occurrence', data=word_occurrences_per_class)
    plt.xlabel('Occurrence')
    plt.ylabel('Class')
    plt.title(f'Distribution of the word {word}')
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


def plot_models_score(score_dict, score_type="F1"):
    # Passing a dict with (key : value) pairs as (model_name : score),
    # function plot a hist comparing all models by score

    models = score_dict.keys()
    values = score_dict.values()
    cmap = plt.get_cmap('Pastel1')

    plt.bar(models, values, color=cmap(range(len(models))))
    plt.xlabel('Models')
    plt.ylabel(f'{score_type} Score')
    plt.title(f'Model Comparison: {score_type} Scores')
    plt.ylim(0, 1)  # Set y-axis limits if desired
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    plt.tight_layout()  # Adjust layout for better fit
    plt.show()


