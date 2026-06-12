"""
GNN model implementations: GCN, GAT, GraphSAGE.

All three architectures share the same overall structure (two message-passing
layers + linear classifier head) and the same forward signature, enabling
plug-and-play comparison.
"""
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv


ArchitectureName = Literal["gcn", "gat", "graphsage"]


class GCN(nn.Module):
    """Two-layer Graph Convolutional Network (Kipf & Welling, 2017)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class GAT(nn.Module):
    """Two-layer Graph Attention Network (Veličković et al., 2018; v2 attention).

    Uses GATv2Conv (Brody et al., 2022), which provides strictly more expressive
    attention than the original GATConv.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        # First layer: multi-head, concat
        self.conv1 = GATv2Conv(
            in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout
        )
        # Second layer: average heads to recover hidden_dim
        self.conv2 = GATv2Conv(
            hidden_dim * heads, hidden_dim, heads=heads, concat=False, dropout=dropout
        )
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        if return_attention:
            x, (edge_idx_1, alpha_1) = self.conv1(
                x, edge_index, return_attention_weights=True
            )
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, (edge_idx_2, alpha_2) = self.conv2(
                x, edge_index, return_attention_weights=True
            )
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            logits = self.classifier(x)
            return logits, (edge_idx_1, alpha_1, edge_idx_2, alpha_2)

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class GraphSAGE(nn.Module):
    """Two-layer GraphSAGE with mean aggregation (Hamilton et al., 2017)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


def build_model(
    architecture: ArchitectureName,
    in_dim: int,
    hidden_dim: int = 64,
    out_dim: int = 2,
    dropout: float = 0.5,
    heads: int = 4,
) -> nn.Module:
    """Factory function that returns a model by architecture name."""
    arch = architecture.lower()
    if arch == "gcn":
        return GCN(in_dim, hidden_dim, out_dim, dropout=dropout)
    if arch == "gat":
        return GAT(in_dim, hidden_dim, out_dim, heads=heads, dropout=dropout)
    if arch == "graphsage":
        return GraphSAGE(in_dim, hidden_dim, out_dim, dropout=dropout)
    raise ValueError(
        f"Unknown architecture: {architecture}. "
        f"Expected one of: gcn, gat, graphsage."
    )
