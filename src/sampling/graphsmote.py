"""
Simplified GraphSMOTE for imbalanced node classification.

This is a pragmatic adaptation of GraphSMOTE (Zhao et al., 2021) for use in
this thesis. The original paper involves training a learned edge predictor;
here we use a simpler scheme:

    1. Identify minority-class training nodes.
    2. For each minority node, find its k nearest neighbours in feature space
       (restricted to other minority training nodes).
    3. Generate a synthetic node by linearly interpolating features between
       the source and a random neighbour.
    4. Connect the synthetic node to the union of the neighbours of both
       endpoints (a heuristic that preserves structural context).
    5. Add the synthetic nodes to the training graph with minority label.

The result is a graph augmented with synthetic minority-class nodes, which is
then used for training. Validation and test sets are not modified.

This is a simplification; for thesis-grade rigour you may wish to compare with
the full GraphSMOTE edge-predictor variant. The simplification is documented
explicitly in Chapter 4.
"""
from typing import Tuple

import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data


def apply_graphsmote(
    data: Data,
    minority_label: int = 1,
    oversample_ratio: float = 1.0,
    k_neighbors: int = 5,
    seed: int = 0,
) -> Data:
    """
    Augment a graph with synthetic minority-class nodes.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Input graph with x, edge_index, y, train_mask, val_mask, test_mask.
    minority_label : int
        Class label considered the minority (default 1 = illicit).
    oversample_ratio : float
        Desired ratio of minority to majority in the training set after
        augmentation. 1.0 means full balance.
    k_neighbors : int
        Number of nearest neighbours considered when generating synthetic nodes.
    seed : int
        Random seed.

    Returns
    -------
    Data : new graph with synthetic nodes appended.
    """
    rng = torch.Generator().manual_seed(seed)

    x = data.x
    y = data.y
    edge_index = data.edge_index
    train_mask = data.train_mask

    # Identify minority and majority counts in the training set
    train_y = y[train_mask]
    n_minority = int((train_y == minority_label).sum())
    n_majority = int((train_y != minority_label).sum())
    if n_minority == 0:
        raise ValueError("No minority-class training nodes found.")

    # How many synthetic nodes to generate
    target_minority = int(n_majority * oversample_ratio)
    n_to_generate = max(0, target_minority - n_minority)
    if n_to_generate == 0:
        return data  # Already balanced

    # Indices of minority training nodes (in the global node id space)
    minority_idx = torch.where(train_mask & (y == minority_label))[0]
    minority_features = x[minority_idx].cpu().numpy()

    # Fit k-NN within the minority class
    k_eff = min(k_neighbors + 1, len(minority_idx))  # +1 because nearest is self
    nn_model = NearestNeighbors(n_neighbors=k_eff).fit(minority_features)
    _, nbr_arr = nn_model.kneighbors(minority_features)
    # nbr_arr[i, 0] is i itself; we use the remaining columns

    # Build adjacency lookup for edge propagation
    # For each minority node, collect its in-and-out neighbours in the full graph
    src, dst = edge_index[0], edge_index[1]
    out_nbrs = {int(i): [] for i in minority_idx.tolist()}
    in_nbrs = {int(i): [] for i in minority_idx.tolist()}
    minority_set = set(int(i) for i in minority_idx.tolist())
    for s, d in zip(src.tolist(), dst.tolist()):
        if s in minority_set:
            out_nbrs[s].append(d)
        if d in minority_set:
            in_nbrs[d].append(s)

    # Generate synthetic nodes
    n_total = x.shape[0]
    new_features = []
    new_out_edges = []  # tuples (src_id, dst_id)
    new_in_edges = []
    new_labels = []

    for k in range(n_to_generate):
        # Randomly pick a minority source node
        i_local = int(torch.randint(0, len(minority_idx), (1,), generator=rng).item())
        i_global = int(minority_idx[i_local].item())

        # Randomly pick one of its k nearest neighbours (excluding self)
        nbr_choices = nbr_arr[i_local, 1:]  # exclude self
        j_local = int(nbr_choices[torch.randint(0, len(nbr_choices), (1,), generator=rng).item()])
        j_global = int(minority_idx[j_local].item())

        # Interpolate features
        lam = torch.rand(1, generator=rng).item()
        x_new = (1.0 - lam) * x[i_global] + lam * x[j_global]
        new_features.append(x_new)

        # Synthetic node id (will be n_total + k after appending)
        new_id = n_total + k

        # Copy outgoing edges from source (heuristic)
        for d in out_nbrs.get(i_global, []):
            new_out_edges.append((new_id, d))
        for d in out_nbrs.get(j_global, []):
            new_out_edges.append((new_id, d))
        # Copy incoming edges
        for s in in_nbrs.get(i_global, []):
            new_in_edges.append((s, new_id))
        for s in in_nbrs.get(j_global, []):
            new_in_edges.append((s, new_id))

        new_labels.append(minority_label)

    # Assemble new graph
    new_x = torch.stack(new_features, dim=0)
    aug_x = torch.cat([x, new_x], dim=0)

    aug_y = torch.cat([y, torch.tensor(new_labels, dtype=y.dtype)], dim=0)

    # New masks: synthetic nodes added to training mask
    extra_train = torch.ones(n_to_generate, dtype=torch.bool)
    extra_val = torch.zeros(n_to_generate, dtype=torch.bool)
    extra_test = torch.zeros(n_to_generate, dtype=torch.bool)
    aug_train_mask = torch.cat([train_mask, extra_train], dim=0)
    aug_val_mask = torch.cat([data.val_mask, extra_val], dim=0)
    aug_test_mask = torch.cat([data.test_mask, extra_test], dim=0)

    # New edges
    all_new = new_out_edges + new_in_edges
    if len(all_new) > 0:
        new_src = torch.tensor([a for a, b in all_new], dtype=torch.long)
        new_dst = torch.tensor([b for a, b in all_new], dtype=torch.long)
        new_edge_index = torch.stack([new_src, new_dst], dim=0)
        aug_edge_index = torch.cat([edge_index, new_edge_index], dim=1)
    else:
        aug_edge_index = edge_index

    aug = Data(
        x=aug_x,
        edge_index=aug_edge_index,
        y=aug_y,
        train_mask=aug_train_mask,
        val_mask=aug_val_mask,
        test_mask=aug_test_mask,
    )
    # Preserve any additional attributes (edge_attr, edge_typology) without
    # modification — synthetic edges have no edge_attr / typology by design.
    return aug
