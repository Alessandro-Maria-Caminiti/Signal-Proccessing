import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    """
    Transformer class for sequence-to-sequence modeling.
    This class implements a Transformer model using PyTorch's `nn.Transformer` module. 
    It includes input and output projections, positional encoding, and a feedforward 
    output layer. The model is designed for tasks such as sequence-to-sequence 
    translation or time-series prediction.
    Attributes:
        input_projection (nn.Linear): Linear layer to project input features to the embedding dimension.
        output_projection (nn.Linear): Linear layer to project target features to the embedding dimension.
        positional_encoding (torch.Tensor): Precomputed positional encoding tensor for adding positional information to input sequences.
        transformer (nn.Transformer): Core Transformer module for encoding and decoding sequences.
        fc_out (nn.Linear): Final linear layer to project the Transformer output to the desired output dimension.
    Methods:
        __init__(input_dim, output_dim, embed_dim, num_heads, num_encoder_layers, num_decoder_layers, ff_dim, dropout):
            Initializes the Transformer model with the specified parameters.
        forward(src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
            Defines the forward pass of the Transformer model.
            Args:
                src (torch.Tensor): Source sequence tensor of shape (batch_size, src_seq_len, input_dim).
                tgt (torch.Tensor): Target sequence tensor of shape (batch_size, tgt_seq_len, output_dim).
                src_mask (torch.Tensor, optional): Source sequence mask for attention (default: None).
                tgt_mask (torch.Tensor, optional): Target sequence mask for attention (default: None).
                memory_mask (torch.Tensor, optional): Memory mask for attention (default: None).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, tgt_seq_len, output_dim).
        _generate_positional_encoding(embed_dim, max_len):
            Generates a positional encoding tensor to add positional information to input sequences.
            Args:
                embed_dim (int): Dimension of the embedding space.
                max_len (int): Maximum sequence length for which positional encoding is generated.
            Returns:
                torch.Tensor: Positional encoding tensor of shape (1, max_len, embed_dim).
    """
    def __init__(self, input_dim, output_dim, embed_dim, num_heads, num_encoder_layers, num_decoder_layers, ff_dim, dropout=0.1):
        super(Transformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)  # Project audio features to embed_dim
        self.output_projection = nn.Linear(output_dim, embed_dim)  # Project target features to embed_dim
        self.positional_encoding = self._generate_positional_encoding(embed_dim, max_len=80).to("cuda")  # Precompute positional encodin

        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ff_dim,
            dropout=dropout
        )

        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src = self.input_projection(src) + self.positional_encoding[:src.size(1), :]
        tgt = self.output_projection(tgt) + self.positional_encoding[:tgt.size(1), :]

        # If input has a channel dimension (batch_size, seq_len, channels, features), merge channels into features
        if src.dim() == 4:
            batch_size,channels, seq_len,  features = src.shape
            src = src.reshape(batch_size, seq_len, channels * features)
        if tgt.dim() == 4:
            batch_size,channels, seq_len, features = tgt.shape
            tgt = tgt.reshape(batch_size, seq_len, channels * features)

        src = src.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        tgt = tgt.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        output = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask)
        output = self.fc_out(output.permute(1, 0, 2))  # (batch_size, seq_len, output_dim)

        return output

    def _generate_positional_encoding(self, embed_dim, max_len):
        positional_en = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        positional_en[:, 0::2] = torch.sin(position * div_term)
        positional_en[:, 1::2] = torch.cos(position * div_term)
        return positional_en.unsqueeze(0)
    