import logging
import math
import torch
import torch.nn as nn

class SimpleLinearProjectionAttention(nn.Module):
    def __init__(
        self,
        attn_input_img_size: int,
        text_features_size: int,
        beta=-1.0,
        norm_type: str = "batch_norm",
    ):
        """
        A simple linear layer that only takes an image embedding as input.

        Args:
            attn_input_img_size (_type_): input dim
            text_features_size (_type_): embed dim
            normalization (str): how to normalize the attention scores as probabilities, options: softmax or sigmoid
            norm_type (str): normalize inputs. values: batch_norm, layer_norm, or None
            beta: -1 (default) means we learn a beta param, else we use the hardcoded value that is specified.
        """
        super().__init__()
        if beta == -1:
            # Learn beta:
            self.beta = torch.nn.Parameter(torch.rand(1).squeeze() + 0.7)
        else:
            # Hardcode beta:
            self.beta = torch.nn.Parameter(torch.tensor(beta), requires_grad=False)
        logging.info(f"Using attn_beta = {self.beta}, type = SimpleLinearProjectionAttention")

        self.norm = (
            torch.nn.BatchNorm1d(attn_input_img_size)
            if norm_type == "batch_norm"
            else nn.LayerNorm(attn_input_img_size)
            if norm_type == "layer_norm"
            else None
        )
        self.fc = torch.nn.Linear(attn_input_img_size, text_features_size)

    def forward(self, img_embedding: torch.Tensor):
        x = img_embedding
        if self.norm is not None:
            x = self.norm(img_embedding)
        attention_scores = self.fc(x)
        return attention_scores