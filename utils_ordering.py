import torch
from torch_sparse import SparseTensor


def random_node_permutation(data):
    """
    Randomly permutes the node ordering while preserving the graph structure.

    Given a permutation P, this function constructs

        X' = P X
        Y' = P Y
        A' = P A P^T

    so that the graph remains identical but the node processing
    sequence changes.
    """

    N = data.num_nodes

    # Random permutation
    g = torch.Generator(device="cpu")
    g.manual_seed(42)

    perm = torch.randperm(N, generator=g)   

    # Inverse permutation
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(N)

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------
    data.x = data.x[perm]

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------
    data.y = data.y[perm]

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------
    for mask_name in ["train_mask", "val_mask", "test_mask"]:

        if not hasattr(data, mask_name):
            continue

        mask = getattr(data, mask_name)

        if mask is None:
            continue

        if mask.dim() == 1:
            setattr(data, mask_name, mask[perm])
        else:
            setattr(data, mask_name, mask[perm, :])

    # --------------------------------------------------------
    # Edge index
    # --------------------------------------------------------
    row = data.edge_index[0]
    col = data.edge_index[1]

    row = inv_perm[row]
    col = inv_perm[col]

    data.edge_index = torch.stack([row, col], dim=0)

    # --------------------------------------------------------
    # Remove cached adjacency if present
    # --------------------------------------------------------
    if hasattr(data, "adj_t"):
        delattr(data, "adj_t")

    return data