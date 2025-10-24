import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPModel
import pdb

class TinyTransformerTextEncoder(nn.Module):
    def __init__(self, vocab_size, seq_len=30, embed_dim=64, output_dim=32, n_layers=1, n_heads=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))  # (1, 30, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, text_tokens):
        input_ids = text_tokens["input_ids"]  # (B, 30)
        attention_mask = text_tokens["attention_mask"]  # (B, 30)

        x = self.embedding(input_ids)  # (B, 30, embed_dim)
        x = x + self.pos_embedding                

        key_padding_mask = (attention_mask == 0)  # (B, 30)

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (B, 30, embed_dim)
        x = x[:, 0, :]

        out = self.output_proj(x)  # (B, output_dim)
        return out
    
    
def get_text_model(task_name, language_emb_model):
    if language_emb_model == "clip":
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
        vocab_size = 49408  # For CLIP's tokenizer
        text_model = TinyTransformerTextEncoder(vocab_size=vocab_size)
    else:
        tokenizer = None
        text_model = None

    max_length = 30
    return text_model, tokenizer, max_length


def extract_text_features(text_model, text_tokens, language_emb_model):
    if language_emb_model == "clip":
        text_latents = text_model(text_tokens)
    else:
        pdb.set_trace()
    return text_latents
