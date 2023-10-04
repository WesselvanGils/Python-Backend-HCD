import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_count, num_classes):
        super().__init__()

        self.stack = nn.Sequential()
        self.stack.add_module("fcfirst", nn.Linear(input_size, hidden_size))
        self.stack.add_module(f"relufirst", nn.ReLU())

        for i in range(hidden_count):
            self.stack.add_module(
                f"fc{i}", nn.Linear(hidden_size, hidden_size))
            self.stack.add_module(f"relu{i}", nn.ReLU())

        self.stack.add_module("fclast", nn.Linear(hidden_size, num_classes))
        self.stack.add_module("last", nn.Softmax(dim=0))

    def forward(self, x):
        out = self.stack(x)
        return out
