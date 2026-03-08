import torch
import torch.nn as nn


class BlockDiagonalGRU(nn.Module):
    """GRU implemented with block-diagonal recurrent weights.

    This enables large number of memory units without quadratic increase in parameters/FLOPs.
    """

    def __init__(self, input_dim, hidden_dim, n_blocks):
        super().__init__()

        assert hidden_dim % n_blocks == 0
        self.n_blocks = n_blocks
        self.block_size = hidden_dim // n_blocks

        assert input_dim % n_blocks == 0
        self.input_size = input_dim
        self.input_block_size = input_dim // n_blocks

        self.cells = nn.ModuleList(
            [
                nn.GRUCell(self.input_block_size, self.block_size)
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x, h):
        """
        Args:
            x (batch, input_dim): input
            h (batch, hidden_dim): prior hidden state

        Returns:
            (batch, hidden_dim): next hidden state
        """
        # split input and hidden vectors into `n_blocks` chunks
        x_blocks = x.chunk(self.n_blocks, dim=-1)
        h_blocks = h.chunk(self.n_blocks, dim=-1)

        # process each chunk in its respective GRU cell
        new_h_blocks = [
            cell(x_i, h_i) for cell, x_i, h_i in zip(self.cells, x_blocks, h_blocks)
        ]

        # concatenate chunk outputs of each GRU cell back together
        new_h = torch.cat(new_h_blocks, dim=-1)
        return new_h
