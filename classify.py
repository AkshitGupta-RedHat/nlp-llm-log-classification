import pandas as pd
from pandas.core.array_algos.replace import should_use_regex

from processor_regex import classify_with_regex
from processor_bert import classify_with_bert
from processor_llm import classify_with_llm



def classify(logs):
    labels = []
    for source, log_msg in logs:
        label = classify_logs(source, log_msg)
        labels.append(label)
    return labels

def classify_logs(source, log_message):
    if source == "LegacyCRM":
        label = classify_with_llm(log_message)
    else:
        label = classify_with_regex(log_message)
        if label is None:
            label = classify_with_bert(log_message)

    return label

def classify_csv(input_file_path):
    df = pd.read_csv(input_file_path)

    df["target_label"] = classify(list(zip(df["source"], df["log_message"])))
    output_file_path = "resources\output.csv"
    df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    classification_results = classify_csv("resources/test.csv")

