import pickle
from collections import defaultdict

class NGramModel:
    def __init__(self, n=3):
        self.n = n
        self.model = defaultdict(list)

    def train(self, text):
        """Train an N-gram model on the given text."""
        tokens = text.lower().strip().split()  # Normalize text
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i + self.n - 1])
            next_word = tokens[i + self.n - 1]
            self.model[context].append(next_word)

    def save_model(self, file_path):
        """Save the trained model to a file."""
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)


def main():
    # Load dataset
    with open("dataset.txt", "r") as file:
        data = file.read()

    # Train N-gram model
    ngram_model = NGramModel(n=3)
    ngram_model.train(data)

    # Save the trained model
    ngram_model.save_model("model/next_word_model.pkl")
    print("Model training complete and saved to next_word_model.pkl")

if __name__ == "__main__":
    main()