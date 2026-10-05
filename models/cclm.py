import math

import torch
from torch import nn


class DirectionalCrissCrossMatch(nn.Module):
    """
    Target-oriented cross-modal correspondence matching.

    Expected spatial layouts:
        target    : 40 x 40 = 1600 tokens
        auxiliary : 20 x 20 =  400 tokens

    For each target location (rt, ct), correspondence candidates
    are restricted to the row and column passing through the
    corresponding auxiliary location (rt // 2, ct // 2).

    The module updates only the target representation.
    Auxiliary tokens are used only as keys/values.
    """

    def __init__(
        self,
        target_dim=1024,
        auxiliary_dim=4096,
        match_dim=256,
        target_hw=(40, 40),
        auxiliary_hw=(20, 20),
    ):
        super().__init__()

        self.target_dim = target_dim
        self.auxiliary_dim = auxiliary_dim
        self.match_dim = match_dim

        self.target_h, self.target_w = target_hw
        self.aux_h, self.aux_w = auxiliary_hw

        assert self.target_h % self.aux_h == 0
        assert self.target_w % self.aux_w == 0

        self.scale_h = self.target_h // self.aux_h
        self.scale_w = self.target_w // self.aux_w

        self.to_q = nn.Linear(target_dim, match_dim, bias=False)
        self.to_k = nn.Linear(auxiliary_dim, match_dim, bias=False)
        self.to_v = nn.Linear(auxiliary_dim, match_dim, bias=False)

        self.to_out = nn.Linear(match_dim, target_dim, bias=False)

        candidate_index = self._build_candidate_index()

        # Fixed spatial topology; not a trainable parameter.
        self.register_buffer(
            "candidate_index",
            candidate_index,
            persistent=False,
        )

        # Visualization/debug hooks. These are populated only when requested.
        self.last_attention = None
        self.last_matched_aux = None

    def _build_candidate_index(self):
        """
        Build [Nt, Nc] auxiliary indices.

        Nt = 40*40 = 1600 target locations.
        Nc = 20+20-1 = 39 auxiliary candidates.
        """
        candidates = []

        for rt in range(self.target_h):
            ra = rt // self.scale_h

            for ct in range(self.target_w):
                ca = ct // self.scale_w

                row = [
                    ra * self.aux_w + j
                    for j in range(self.aux_w)
                ]

                # Exclude the center from the column because it is
                # already included in the row.
                col = [
                    i * self.aux_w + ca
                    for i in range(self.aux_h)
                    if i != ra
                ]

                candidates.append(row + col)

        index = torch.tensor(candidates, dtype=torch.long)

        expected_nt = self.target_h * self.target_w
        expected_nc = self.aux_w + self.aux_h - 1

        assert index.shape == (expected_nt, expected_nc)

        return index

    def forward(self, target, auxiliary, return_visuals=False):
        """
        Args:
            target:
                [B, Nt, target_dim]
            auxiliary:
                [B, Na, auxiliary_dim]
            return_visuals:
                if True, also return correspondence evidence.

        Returns:
            matched_aux:
                [B, Nt, target_dim]

            visuals (optional):
                attention   : [B, Nt, 39]
                matched_aux : [B, Nt, target_dim]
                candidate_index : [Nt, 39]
        """
        B, Nt, Ct = target.shape
        Ba, Na, Ca = auxiliary.shape

        assert B == Ba
        assert Nt == self.target_h * self.target_w
        assert Na == self.aux_h * self.aux_w
        assert Ct == self.target_dim
        assert Ca == self.auxiliary_dim

        q = self.to_q(target)       # [B, Nt, D]
        k = self.to_k(auxiliary)    # [B, Na, D]
        v = self.to_v(auxiliary)    # [B, Na, D]

        # Gather the 39 structured auxiliary candidates for every
        # target query.
        #
        # k[:, candidate_index, :]
        # -> [B, Nt, 39, D]
        candidate_k = k[:, self.candidate_index, :]
        candidate_v = v[:, self.candidate_index, :]

        scores = (
            q.unsqueeze(2) * candidate_k
        ).sum(dim=-1) / math.sqrt(self.match_dim)

        attention = torch.softmax(scores, dim=-1)

        matched = (
            attention.unsqueeze(-1) * candidate_v
        ).sum(dim=2)

        matched_aux = self.to_out(matched)

        if return_visuals:
            visuals = {
                "attention": attention.detach(),
                "matched_aux": matched_aux.detach(),
                "candidate_index": self.candidate_index.detach(),
            }
            return matched_aux, visuals

        return matched_aux
