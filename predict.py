import pickle
from collections import defaultdict
import threading

class NGramModel:
    def __init__(self, n=3):
        self.n = n
        self.model = defaultdict(list)

    def load_model(self, file_path):
        """Load the trained model from a file."""
        with open(file_path, "rb") as f:
            self.model = pickle.load(f)

    def predict_next_word(self, context):
        """Predict the next word based on the context."""
        context = tuple(context.lower().strip().split())  # Normalize the context
        if context in self.model:
            return self.model[context]
        else:
            return ["No prediction available"]


class SingletonModelLoader:
    """
    Singleton class to ensure the model is loaded only once into memory.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_path):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SingletonModelLoader, cls).__new__(cls)
                cls._instance._initialize(model_path)
            return cls._instance

    def _initialize(self, model_path):
        """Private method to initialize the model loader."""
        self.model = NGramModel(n=3)
        self.model.load_model(model_path)

    def predict(self, context):
        """Perform prediction using the loaded model."""
        if self.model is None:
            raise ValueError("Model is not loaded.")
        return self.model.predict_next_word(context)


def main():
    # Initialize the SingletonModelLoader
    model_path = "model/next_word_model.pkl"
    model_loader = SingletonModelLoader(model_path)

    # Example context for inference
    context = "how are"
    prediction = model_loader.predict(context)

    print(f"Context: {context}")
    print(f"Predicted next word(s): {prediction}")


if __name__ == "__main__":
    main()