import math

import torch
import torch.nn as nn


def block_sizes(input_size: int, n_blocks: int) -> list[int]:
    """Calculate block sizes given input vector size and number of blocks.

    If `n_blocks` does not evenly divide `input_size`, then the last block
    will be of lesser size.

    Args:
        input_size (int): size of input vector.
        n_blocks (int): number of blocks to split vector into.

    Returns:
        list[int]: list of block sizes.
    """
    if input_size < n_blocks:
        raise ValueError(f"Cannot split given input_size {input_size} into {n_blocks} blocks.")
    if input_size % n_blocks == 0:
        block_size = input_size // n_blocks
        return n_blocks * [block_size]
    else:
        # first (n - 1) blocks will have more units
        block_size = math.ceil(input_size / n_blocks)
        blocks = (n_blocks - 1) * [block_size]
        # last block will contain the remaining units
        last_block_size = input_size - (n_blocks - 1) * block_size
        blocks.append(last_block_size)
        return blocks


class BlockDiagonalGRU(nn.Module):
    """GRU implemented with block-diagonal recurrent weights.

    This enables large number of memory units without quadratic increase in parameters/FLOPs.
    """

    def __init__(self, input_size, recurrent_size, n_blocks):
        super().__init__()
        self.n_blocks = n_blocks
        self.recurrent_block_sizes = block_sizes(recurrent_size, n_blocks)
        self.input_block_sizes = block_sizes(input_size, n_blocks)
        self.cells = nn.ModuleList(
            [nn.GRUCell(i_size, h_size) for i_size, h_size in zip(self.input_block_sizes, self.recurrent_block_sizes)]
        )
        self.layer_norm = nn.LayerNorm(recurrent_size)

    def forward(self, x, h):
        """
        Args:
            x (batch, input_size): input tensor
            h (batch, recurrent_size): recurrent state tensor

        Returns:
            (batch, recurrent_size): next recurrent state tensor
        """
        # split input and hidden vectors into `n_blocks` chunks
        x_blocks = x.chunk(self.n_blocks, dim=-1)
        h_blocks = h.chunk(self.n_blocks, dim=-1)
        # process each chunk in its respective GRU cell
        new_h_blocks = [cell(x_i, h_i) for cell, x_i, h_i in zip(self.cells, x_blocks, h_blocks)]
        # concatenate chunk outputs of each GRU cell back together
        new_h = torch.cat(new_h_blocks, dim=-1)
        return self.layer_norm(new_h)
