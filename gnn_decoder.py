"""
Pretrained graph-neural-network decoder of Lange et al., Phys. Rev. Research 7,
023181 (2025), used here to extract per-shot logits.

The network weights are the released checkpoints (gnn_models/d{d}_d_t_{r}.pt,
fetch_gnn_models.py); the architecture below is copied from the authors'
repository (https://github.com/LangeMoritz/GNN_decoder, src/gnn_models.py;
MIT License, Copyright (c) 2023 Moritz Lange, see THIRD_PARTY_NOTICES.md) so
that the checkpoints load unchanged.

Graph construction. Each detection event is a node with features
(is_X, is_Z, x, y, t); every node is connected to its M = 10 nearest
neighbours (Euclidean distance on the detector grid), with edge attribute
1 / distance^2. Because detector coordinates are integers, equidistant
neighbours are common and the tie-breaking rule changes the graph. The
datasets of the paper were produced on GPU with torch_cluster.knn_graph
(loop=False), whose CUDA kernel keeps the LOWER node index among equidistant
candidates. _knn_graph() below reproduces that rule exactly (stable sort of
squared distances), so this module gives the same decisions and logits as the
original GPU runs up to floating-point rounding (verified on the original
(5, 5) stream: identical decisions, max |Delta| difference 9e-5 dB).
A plain torch.topk does NOT reproduce them.

Sign convention (identical to post_selection.MWPMSoftDecoder):
    gap = -logit  [nats];   gap > 0  <=>  prediction 0.
Trivial syndromes (no detection event) bypass the network, as in the authors'
pipeline: prediction 0 and, as confidence, the MWPM gap of the trivial
syndrome (the maximal matching gap).
"""

import numpy as np
import stim
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GraphConv, global_mean_pool

from post_selection import MWPMSoftDecoder


class GNN7(nn.Module):
    """GNN_7 of Lange et al. Layer names and shapes must match the checkpoints."""

    def __init__(self, hidden_channels_GCN=(32, 128, 256, 512, 512, 256, 256),
                 hidden_channels_MLP=(256, 128, 64), num_node_features=5, num_classes=1):
        super().__init__()
        channels = [num_node_features, *hidden_channels_GCN]
        self.graph_layers = nn.ModuleList(
            [GraphConv(c_in, c_out) for c_in, c_out in zip(channels[:-1], channels[1:])])
        channels = [hidden_channels_GCN[-1], *hidden_channels_MLP]
        self.dense_layers = nn.ModuleList(
            [nn.Linear(c_in, c_out) for c_in, c_out in zip(channels[:-1], channels[1:])])
        self.output_layer = nn.Linear(hidden_channels_MLP[-1], num_classes)

    def forward(self, x, edge_index, batch, edge_attr):
        for layer in self.graph_layers:
            x = F.relu(layer(x, edge_index, edge_attr), inplace=True)
        x = global_mean_pool(x, batch)
        for layer in self.dense_layers:
            x = F.relu(layer(x), inplace=True)
        return self.output_layer(x)  # raw logit; sigmoid(logit) ~ P(logical flip)


class GNNDecoder:
    """Pretrained GNN decoder for the rotated surface code, rotated_memory_z."""

    ARCH_SMALL = dict(hidden_channels_GCN=(32, 128, 256, 512, 512, 256, 256),
                      hidden_channels_MLP=(256, 128, 64))
    ARCH_LARGE = dict(hidden_channels_GCN=(32, 128, 256, 512, 512, 512, 512),
                      hidden_channels_MLP=(512, 256, 128, 64, 32, 16))
    M_NEAREST_NODES = 10
    POWER = 2

    def __init__(self, dem: stim.DetectorErrorModel, distance: int, rounds: int,
                 checkpoint_path: str, device: str = "cpu", chunk_size: int = 10_000):
        self.distance = distance
        self.rounds = rounds
        self.device = torch.device(device)
        self.chunk_size = chunk_size

        arch = self.ARCH_LARGE if distance == 9 else self.ARCH_SMALL  # d = 9 uses the larger net
        self.model = GNN7(**arch).to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model.load_state_dict(state)
        self.model.eval()

        self._init_circuit_geometry()
        _, trivial_gap = MWPMSoftDecoder(dem).decode_with_gap(
            np.zeros((1, dem.num_detectors), dtype=np.uint8))
        self.trivial_gap = float(trivial_gap[0])

    # --- syndrome -> graph ---------------------------------------------------

    def _init_circuit_geometry(self):
        """Detector coordinates and X/Z stabiliser mask (depend only on d, r)."""
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z", rounds=self.rounds, distance=self.distance,
            after_clifford_depolarization=1e-3, before_round_data_depolarization=1e-3,
            before_measure_flip_probability=1e-3, after_reset_flip_probability=1e-3)
        coords = np.array(list(circuit.get_detector_coordinates().values()))
        coords[:, :2] = coords[:, :2] / 2  # stim uses a half-integer grid
        self._detector_coords = coords.astype(np.uint8)
        size = self.distance + 1
        sx = np.zeros((size, size), dtype=np.uint8)
        sx[::2, 1:size - 1:2] = 1
        sx[1::2, 2::2] = 1
        sz = np.rot90(sx) * 3
        self._syndrome_mask = np.dstack([sx + sz] * (self.rounds + 1))  # 1 = X, 3 = Z

    def _syndromes_to_3d(self, syndromes):
        mask = np.repeat(self._syndrome_mask[None, ...], syndromes.shape[0], 0)
        grid = np.zeros_like(mask)
        grid[:, self._detector_coords[:, 1], self._detector_coords[:, 0],
             self._detector_coords[:, 2]] = syndromes
        grid[np.nonzero(grid)] = mask[np.nonzero(grid)]
        return grid

    @staticmethod
    def _knn_graph(pos, k, batch):
        """k-nearest-neighbour graph per shot, torch_cluster CUDA tie-breaking.

        Returns edge_index (2, E) with row 0 = neighbour (source), row 1 = node
        (target), as torch_cluster.knn_graph(..., loop=False, flow='source_to_target').
        """
        device = pos.device
        _, counts = torch.unique_consecutive(batch, return_counts=True)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])
        chunks = []
        for i in range(len(counts)):
            s, e = offsets[i].item(), offsets[i + 1].item()
            n = e - s
            if n < 2:
                continue
            p = pos[s:e]
            diff = p.unsqueeze(0) - p.unsqueeze(1)
            d2 = (diff * diff).sum(dim=-1)
            d2.fill_diagonal_(float("inf"))
            k_eff = min(k, n - 1)
            nbr = torch.sort(d2, dim=1, stable=True).indices[:, :k_eff]
            node = torch.arange(n, device=device).unsqueeze(1).expand(-1, k_eff)
            chunks.append(torch.stack([nbr.reshape(-1), node.reshape(-1)], dim=0) + s)
        if not chunks:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.cat(chunks, dim=1)

    def _build_batch_graph(self, syndromes):
        grid = self._syndromes_to_3d(syndromes).astype(np.float32)
        defect_inds = np.nonzero(grid)
        defects = grid[defect_inds]
        defect_inds = np.transpose(np.array(defect_inds))
        node_features = np.zeros((defects.shape[0], 6), dtype=np.float32)  # is_X, is_Z, shot, x, y, t
        x_def, z_def = defects == 1, defects == 3
        node_features[x_def, 0] = 1
        node_features[x_def, 2:] = defect_inds[x_def]
        node_features[z_def, 1] = 1
        node_features[z_def, 2:] = defect_inds[z_def]
        x = torch.tensor(node_features[:, [0, 1, 3, 4, 5]]).to(self.device)
        batch = torch.tensor(node_features[:, 2]).long().to(self.device)
        pos = x[:, 2:]
        edge_index = self._knn_graph(pos, self.M_NEAREST_NODES, batch)
        dist = torch.sqrt(((pos[edge_index[0]] - pos[edge_index[1]]) ** 2).sum(dim=1, keepdim=True))
        edge_attr = 1.0 / dist ** self.POWER
        return x, edge_index, batch, edge_attr

    # --- decoding ------------------------------------------------------------

    @torch.no_grad()
    def decode_with_gap(self, syndromes):
        """Return (predictions uint8, gaps float64 [nats]); gap = -logit."""
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        n_shots = syndromes.shape[0]
        predictions = np.zeros(n_shots, dtype=np.uint8)
        gaps = np.zeros(n_shots, dtype=np.float64)
        is_trivial = syndromes.sum(axis=1) == 0
        gaps[is_trivial] = self.trivial_gap
        nontrivial = np.flatnonzero(~is_trivial)
        for start in range(0, len(nontrivial), self.chunk_size):
            idx = nontrivial[start:start + self.chunk_size]
            x, edge_index, batch, edge_attr = self._build_batch_graph(syndromes[idx])
            logits = self.model(x, edge_index, batch, edge_attr).squeeze(-1).cpu().numpy()
            gaps[idx] = -logits.astype(np.float64)
            predictions[idx] = (logits > 0).astype(np.uint8)
        return predictions, gaps
