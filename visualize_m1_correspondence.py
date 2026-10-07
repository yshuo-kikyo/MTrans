from pathlib import Path
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from models.cclm import DirectionalCrissCrossMatch


def restore_auxiliary_attention(
    attention,
    candidate_index,
    query_index,
    auxiliary_hw=(20, 20),
):
    """
    Restore one target query's structured attention from [39]
    into the native auxiliary 20x20 spatial grid.

    Positions outside the directional candidate support remain zero.
    """
    aux_h, aux_w = auxiliary_hw

    assert attention.ndim == 2
    assert candidate_index.ndim == 2
    assert attention.shape == candidate_index.shape

    weights = attention[query_index]
    indices = candidate_index[query_index]

    spatial = torch.zeros(
        aux_h * aux_w,
        dtype=weights.dtype,
        device=weights.device,
    )

    spatial[indices] = weights

    return spatial.reshape(aux_h, aux_w)


def expected_support(
    query_index,
    target_hw=(40, 40),
    auxiliary_hw=(20, 20),
):
    """
    Return the expected auxiliary criss-cross support mask
    for one target query.
    """
    target_h, target_w = target_hw
    aux_h, aux_w = auxiliary_hw

    rt = query_index // target_w
    ct = query_index % target_w

    scale_h = target_h // aux_h
    scale_w = target_w // aux_w

    ra = rt // scale_h
    ca = ct // scale_w

    mask = torch.zeros(
        aux_h,
        aux_w,
        dtype=torch.bool,
    )

    mask[ra, :] = True
    mask[:, ca] = True

    return mask, (rt, ct), (ra, ca)


def save_figure(
    spatial_attention,
    support_mask,
    target_rc,
    auxiliary_rc,
    output_path,
):
    attn = spatial_attention.detach().cpu().numpy()
    support = support_mask.detach().cpu().numpy()

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(attn, interpolation="nearest")
    ax1.scatter(
        [auxiliary_rc[1]],
        [auxiliary_rc[0]],
        marker="x",
        s=100,
    )
    ax1.set_title(
        f"Directional attention\n"
        f"target={target_rc}, aux center={auxiliary_rc}"
    )
    ax1.set_xlabel("Auxiliary column")
    ax1.set_ylabel("Auxiliary row")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(support.astype(np.float32), interpolation="nearest")
    ax2.scatter(
        [auxiliary_rc[1]],
        [auxiliary_rc[0]],
        marker="x",
        s=100,
    )
    ax2.set_title("Criss-cross candidate support (39 positions)")
    ax2.set_xlabel("Auxiliary column")
    ax2.set_ylabel("Auxiliary row")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query-row",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--query-col",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualizations/m1_directional_attention_test.png",
    )

    args = parser.parse_args()

    target_h = 40
    target_w = 40
    aux_h = 20
    aux_w = 20

    assert 0 <= args.query_row < target_h
    assert 0 <= args.query_col < target_w

    query_index = (
        args.query_row * target_w
        + args.query_col
    )

    torch.manual_seed(42)

    matcher = DirectionalCrissCrossMatch()
    matcher.eval()

    target = torch.randn(
        1,
        target_h * target_w,
        1024,
    )

    auxiliary = torch.randn(
        1,
        aux_h * aux_w,
        4096,
    )

    with torch.no_grad():
        _, visuals = matcher(
            target,
            auxiliary,
            return_visuals=True,
        )

    attention = visuals["attention"][0]
    candidate_index = visuals["candidate_index"]

    spatial_attention = restore_auxiliary_attention(
        attention,
        candidate_index,
        query_index,
        auxiliary_hw=(aux_h, aux_w),
    )

    support_mask, target_rc, auxiliary_rc = expected_support(
        query_index,
        target_hw=(target_h, target_w),
        auxiliary_hw=(aux_h, aux_w),
    )

    actual_support = spatial_attention > 0

    print("===== QUERY =====")
    print("query index       :", query_index)
    print("target location   :", target_rc)
    print("auxiliary center  :", auxiliary_rc)

    print("\n===== TOPOLOGY =====")
    print("candidate count   :", candidate_index[query_index].numel())
    print("expected support  :", support_mask.sum().item())
    print("actual support    :", actual_support.sum().item())

    assert candidate_index[query_index].numel() == 39
    assert support_mask.sum().item() == 39
    assert actual_support.sum().item() == 39

    assert torch.equal(
        actual_support.cpu(),
        support_mask,
    )

    attention_sum = spatial_attention.sum().item()

    print("\n===== ATTENTION =====")
    print("spatial sum       :", attention_sum)
    print(
        "original sum      :",
        attention[query_index].sum().item(),
    )

    assert torch.allclose(
        spatial_attention.sum(),
        attention[query_index].sum(),
        atol=1e-6,
    )

    outside = spatial_attention[~support_mask]

    assert torch.count_nonzero(outside).item() == 0

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_figure(
        spatial_attention,
        support_mask,
        target_rc,
        auxiliary_rc,
        output_path,
    )

    print("\nfigure:", output_path)
    print(
        "\n===== M1 SPATIAL CORRESPONDENCE "
        "VISUALIZATION PASSED ====="
    )


if __name__ == "__main__":
    main()
