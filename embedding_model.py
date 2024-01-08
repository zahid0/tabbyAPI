import torch
from transformers import AutoModel, AutoTokenizer


class EmbeddingModelContainer:
    def __init__(self, embedding_model_name: str, max_length: int):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.model = AutoModel.from_pretrained(embedding_model_name).to(device)
        self.max_length = max_length
        self.device = device
        self.model_name = embedding_model_name

    def get_model_name(self):
        return self.model_name

    def get_embeddings(self, sentences):
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        # move tokenizer inputs to device
        encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        context_layer: "torch.Tensor" = model_output[0]
        # CLS Pooling
        if len(context_layer.shape) == 3:
            embeddings = context_layer[:, 0]
        elif len(context_layer.shape) == 2:
            embeddings = context_layer[0]
        else:
            raise NotImplementedError(f"Unhandled shape {embeddings.shape}.")
        return embeddings.tolist()
