from sentence_transformers import SentenceTransformer
import joblib

# Load the sentence transformer model and the pre-trained classifier
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Example sentence transformer model
clf_model = joblib.load("models/clf_model.joblib")


def classify_with_bert(log_messages):

    # Generate sentence embeddings for log messages
    embeddings = embedder.encode(log_messages)

    probabilities = clf_model.predict_proba([embeddings])[0]

    if max(probabilities) < 0.5:
        # Predict labels using the pre-trained model
        return "unclassified"
    predicted_class = clf_model.predict([embeddings])[0]

    return predicted_class



if __name__ == "__main__":
    log_messages = ["This is a test log message.", "This is another test log message."]
    logs = [
        # "User User\d+ logged in",
  #  "Backup (started|ended) at .*",
  #  "Backup completed successfully",
    "System updated version .*",
  #  "File .* uploaded successfully",
 #   "Disk cleanup completed successfully",
  #  "System reboot initiated by user",
    #"Account with ID .* created .*"
    ]

    for log in logs:
        label = classify_with_bert(log)
        print(f"Log message: {log}, Label: {label}")

