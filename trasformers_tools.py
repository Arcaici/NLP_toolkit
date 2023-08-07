import time
import numpy as np
import pandas as pd
from datasets import ClassLabel
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from umap import UMAP

def plot_number_of_features(text_column):
    # this function plot a subplot compose of 2 barchart, where each barchart show the distribution of
    # token number of text feature respect to bert and pegasus token techniques

    model_bert = "nlpaueb/legal-bert-small-uncased"
    tokenizer_bert = AutoTokenizer.from_pretrained(model_bert)
    model_pegasus = "nsi319/legal-pegasus"
    tokenizer_pegasus = AutoTokenizer.from_pretrained(model_pegasus)

    # plotting sentences lenghth distribution (in terms of token)
    bert_len = [len(tokenizer_bert.encode(s)) for s in text_column]
    pagasus_len = [len(tokenizer_pegasus.encode(s)) for s in text_column]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    axes[0].hist(bert_len, bins=100, color="C0", edgecolor="C0")
    axes[0].axvline(x=512, ymin=0, ymax=1, linestyle="--", color="C1",
               label="Maximum sequence length")
    axes[0].set_title("Bert Token Length")
    axes[0].set_xlabel("Length")
    axes[0].set_ylabel("Count")
    axes[1].hist(pagasus_len, bins=100, color="C0", edgecolor="C0")
    axes[1].axvline(x=1024, ymin=0, ymax=1, linestyle="--", color="C1",
                    label="Maximum sequence length")
    axes[1].set_title("Pegasus Token Length")
    axes[1].set_xlabel("Length")
    plt.tight_layout()
    plt.show()

def plot_label_distribution_pattern(X, y, labels_list):
    #This function at first project onto lower dimention (2D) the features
    # and then plot a hexagonal binning plot over all classes showing
    x_scaled = MinMaxScaler().fit_trasform(X)
    mapper = UMAP(n_components=2, metric= "cosine").fit(x_scaled)
    df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
    df_emb["label"] = y


    fig, axes = plt.subplots(5, 4, figsize=(10, 7))
    axes = axes.flatten()
    cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'Greys', 'Purples']
    labels = labels_list

    for i, (label, cmap) in enumerate(zip(labels, cmaps)):
        df_emb_sub = df_emb.query(f"label == {i}")
        axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap,
                       gridsize=20, linewidths=(0,))
        axes[i].set_title(label)
        axes[i].set_xticks([]), axes[i].set_yticks([])

    plt.tight_layout()
    plt.show()

def pegasus_summary(batch_samples):
    # This function take in input a batch of samples and return the summary of each sample.
    # The summary length is set to 400 token length, because the output summary will be used as bert tokenizer input
    # LLM used: legal-pegasus
    # It will be better to call this function with model anf tokenizer already define inside the main code

    summary = ""
    # summary
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_ckpt = "nsi319/legal-pegasus"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
    input_tokenized = tokenizer.encode(batch_samples, return_tensors='pt', max_length=1024, truncation=True).to(device)
    with torch.no_grad():
        summary_ids = model.generate(input_tokenized,
                                     num_beams=9,
                                     no_repeat_ngram_size=3,
                                     length_penalty=2.0,
                                     min_length=150,
                                     max_length=400,
                                     early_stopping=True)

    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
    return {"text": summary}

def bert_hidden_state(batch_samples, labels_list):
    # This function take in input a batch of samples and return the last hidden layer of bert.
    # Each example is first tokenized and then is give to bert for extract the last hidden layer
    # for use it as features of the final model.
    # LLM used: legal-bert-small-uncased
    # It will be better to call this function with model anf tokenizer already define inside the main code


    # Transformer settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ckpt = 'nlpaueb/legal-bert-small-uncased'

    # Loading model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt, num_labels=len(labels_list)).to(device)

    print("GPU memory allocated before tokenization", torch.cuda.memory_allocated() / (1024 ** 3))

    # tokenizing samples
    def tokenization(batch):
        # tokenize text
        tokenized_sample = tokenizer(batch["text"], padding=True, truncation=True, max_length=512,
                                     return_tensors="pt").to(device)

        # encode label
        labels = ClassLabel(names=labels_list)
        tokenized_sample["label"] = labels.str2int(batch["label"])

        return tokenized_sample

    data_token = batch_samples.map(tokenization, batched=True)
    print("Done tokenization")
    print("GPU memory allocated after tokenization", torch.cuda.memory_allocated() / (1024 ** 3))
    torch.cuda.empty_cache()

    # mapping inputs
    def extract_hidden_state(batch, device="cpu"):

        # converting all tensor to same device( i use cpu as default because GPU_Ram is not enough)
        model.to(device)
        inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}

        # Retrieving last hidden state
        with torch.no_grad():
            last_hidden_state = model(**inputs).last_hidden_state

        return {"hidden_state": last_hidden_state[:, 0]}

    data_token.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    data_hidden = data_token.map(extract_hidden_state, batched=True)
    print("Done with hidden state")

    return data_hidden

def hiddenState2FeatureMatrix(df) :
    # Giving in input a dataset with "hidden_state" and "label" columns this function
    # return a Feature Matrix X and a Label array y

    # creating a features matrix
    X = np.array(df["hidden_state"])
    y = np.array(df["label"])

    return X, y

def start_recording():
    # Record the start time of a function
    return time.time()

def end_recording(start_time):
    # Record the end time
    end_time = time.time()

    # Execution time in Minutes
    execution_time = end_time - start_time
    print("Execution Time:", execution_time, "seconds")
    print("Execution Time:", (execution_time) / 60, "minutes")