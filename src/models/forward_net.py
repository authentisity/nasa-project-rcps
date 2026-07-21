import torch
import torch.nn as nn


class WingLSTM(nn.Module):
    """Given per-trajectory design/flow conditions, unroll an LSTM over
    time to predict the CL/CD/Cm sequence."""

    def __init__(self, static_size, target_size, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        input_size = static_size + 1  # static conditioning + scalar t

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Linear(hidden_size, target_size)

    def forward(self, static, t, lengths=None):
        # static: (B, static_size)   t: (B, max_len)
        B, max_len = t.shape

        static_seq = static.unsqueeze(1).expand(-1, max_len, -1)
        x = torch.cat([static_seq, t.unsqueeze(-1)], dim=-1)  # (B, max_len, static_size+1)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=max_len)
        else:
            out, _ = self.lstm(x)

        return self.head(out)  # (B, max_len, target_size)
