"""
MIT License

Copyright (c) 2025 Princeton University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import model

class KrakenBlock(torch.nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        n_splits = int(config.n_head/config.n_heads_in_split)
        self.ln_1 = model.LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = model.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = model.CausalSelfAttention(config)
        self.mlp = model.SplitMLP(config)
        
    def forward(self, x, incoming_edges = None):
        
        x = x + self.attn(self.ln_1(x))
        residual = x
        x = x + torch.sum(torch.stack(incoming_edges, dim = 0), dim = 0)
        x = residual + self.mlp(self.ln_2(x))
        return x