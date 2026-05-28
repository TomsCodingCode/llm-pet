from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model, p_drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(p_drop),
        )

    def forward(self, x):
        return self.net(x)
