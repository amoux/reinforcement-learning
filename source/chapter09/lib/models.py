import torch
import torch.nn as nn


class PGN(nn.Module):
    def __init__(self, input_size: int, n_actions: int) -> None:
        super(PGN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, n_actions),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Compute the state inputs into logits (raw output)
        # from the linear layers. Since softmax is not done
        # internally, use softmax to get the log probabilities.
        logits = self.fc(input)
        return logits
