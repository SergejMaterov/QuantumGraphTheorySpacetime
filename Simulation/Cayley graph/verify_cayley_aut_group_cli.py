#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_cayley_aut_group_cli.py

CLI-enabled verification script for colour-preserving automorphisms of a coloured oriented
Cayley graph constructed from a candidate discrete group Gamma_SM = Z_n ⋊ (S3 × Z_k).

Usage examples:
  python verify_cayley_aut_group_cli.py --zN 24 --k3 3 --s1 1 --s2 3 --outdir ./verify_out
  python verify_cayley_aut_group_cli.py --zN 24 --k3 3 --s1_perm "1,0,2" --s2_perm "0,2,1" --outdir ./verify_out

Outputs:
  - verification_report.txt (human readable) in --outdir
  - optionally extra_automorphisms.json if non-left automorphisms are found
"""
import argparse
import json
import os
from collections import deque

# ---------- canonical S3 permutations (as images of 0,1,2) ----------
S3_all = [
    (0, 1, 2),  # id
    (1, 0, 2),  # (01)
    (2, 1, 0),  # (02)
    (0, 2, 1),  # (12)
    (1, 2, 0),  # (012)
    (2, 0, 1),  # (021)
]


# ---------- permutation utilities ----------
def parse_perm(s):
    """
    Parse a comma-separated permutation like "1,0,2" into a tuple (1,0,2).
    Validate that it's a permutation of {0,1,2}.
    """
    parts = [p.strip() for p in s.split(",")]
    try:
        nums = tuple(int(x) for x in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid permutation values: {s!r}")

    if len(nums) != 3:
        raise argparse.ArgumentTypeError(f"Permutation must have exactly 3 entries: {s!r}")
    if set(nums) != {0, 1, 2}:
        raise argparse.ArgumentTypeError(f"Permutation must be a rearrangement of 0,1,2: {s!r}")
    return nums


def compose_perm(p, q):
    """
    Return permutation r = p ∘ q (apply q, then p).
    Represented as tuple r where r[i] = p[q[i]].
    """
    if len(p) != len(q):
        raise ValueError("Permutation lengths must match")
    return tuple(p[q[i]] for i in range(len(q)))


def perm_inverse(p):
    inv = [0] * len(p)
    for i, v in enumerate(p):
        inv[v] = i
    return tuple(inv)


# ---------- Group utilities (configurable zn, k3) ----------
def iter_group(zn, k3):
    """
    Iterate over group elements as triples (z in Z_n, sigma in S3, c in Z_k3).
    sigma is represented as a tuple of 3 ints.
    """
    for z in range(zn):
        for sigma in S3_all:
            for c in range(k3):
                yield (z, sigma, c)


def group_mul_trivial(g1, g2, zn, k3):
    """
    Trivial direct-product multiplication:
      (z1, s1, c1) * (z2, s2, c2) = (z1+z2 mod n, s1∘s2, c1+c2 mod k3)
    (This implements the direct product; replace this if a nontrivial semidirect action is required.)
    """
    z1, s1, c1 = g1
    z2, s2, c2 = g2
    z = (z1 + z2) % zn
    s = compose_perm(s1, s2)
    c = (c1 + c2) % k3
    return (z, s, c)


def group_identity(zn):
    return (0, (0, 1, 2), 0)


def group_invert(g, zn, k3):
    z, s, c = g
    s_inv = perm_inverse(s)
    return ((-z) % zn, s_inv, (-c) % k3)


# ---------- Building Cayley graph ----------
def build_group_and_index(zn, k3, generators):
    """
    Return (GROUP, GROUP_INDEX, ADJ, OUT_MAP)
    - GROUP: list of group elements
    - GROUP_INDEX: mapping element -> index
    - ADJ: adjacency list: for each vertex index, list of (target_index, color, gen_name)
    - OUT_MAP: for each vertex index, dict color -> target_index (fast lookup)
    """
    GROUP = list(iter_group(zn, k3))
    GROUP_INDEX = {g: i for i, g in enumerate(GROUP)}

    def mul(g1, g2):
        return group_mul_trivial(g1, g2, zn, k3)

    N = len(GROUP)
    ADJ = [[] for _ in range(N)]
    OUT_MAP = [{} for _ in range(N)]

    for i, g in enumerate(GROUP):
        for gen_name, gen_elem, col in generators:
            h = mul(g, gen_elem)
            j = GROUP_INDEX.get(h, None)
            if j is None:
                raise RuntimeError(f"Generated element not in group: {h}")
            ADJ[i].append((j, col, gen_name))
            # Enforce unique outgoing edge per colour at each vertex
            if col in OUT_MAP[i] and OUT_MAP[i][col] != j:
                raise RuntimeError(f"Colour collision at vertex {i} colour {col}: {OUT_MAP[i][col]} vs {j}")
            OUT_MAP[i][col] = j

    return GROUP, GROUP_INDEX, ADJ, OUT_MAP


def left_mul_map(h, GROUP, GROUP_INDEX, zn, k3):
    """
    Return list mapping index i (element g) to index of h * g (left multiplication).
    """
    mapping = [None] * len(GROUP)
    for i, g in enumerate(GROUP):
        mapped = group_mul_trivial(h, g, zn, k3)
        mapping[i] = GROUP_INDEX[mapped]
    return mapping


def check_colour_preserving(mapping, ADJ, OUT_MAP):
    """
    mapping: list of length N mapping source index -> image index
    ADJ and OUT_MAP describe the original Cayley graph.
    Returns True if mapping preserves colours and adjacency direction.
    """
    N = len(mapping)
    for u in range(N):
        mu = mapping[u]
        for (v, col, _) in ADJ[u]:
            mv = mapping[v]
            out = OUT_MAP[mu].get(col, None)
            if out is None or out != mv:
                return False
    return True


def build_mapping_from_image_of_identity(target_idx, ID_IDX, ADJ, OUT_MAP):
    """
    Try to construct a full colour-preserving mapping by specifying image(ID) = target_idx
    and propagating along generators (BFS). Return mapping list if successful, else None.
    """
    N = len(ADJ)
    mapping = {ID_IDX: target_idx}
    queue = deque([ID_IDX])

    while queue:
        u = queue.popleft()
        mu = mapping[u]  # image of u
        for (v, col, _) in ADJ[u]:
            # look where mu's outgoing edge of color 'col' goes
            if col not in OUT_MAP[mu]:
                return None
            expected = OUT_MAP[mu][col]
            if v in mapping:
                if mapping[v] != expected:
                    return None
            else:
                mapping[v] = expected
                queue.append(v)

    if len(mapping) < N:
        return None

    # convert to list
    mapping_list = [None] * N
    for src_idx, dst_idx in mapping.items():
        mapping_list[src_idx] = dst_idx

    # sanity: no None and all images distinct
    if any(m is None for m in mapping_list):
        return None
    if len(set(mapping_list)) != N:
        return None

    if not check_colour_preserving(mapping_list, ADJ, OUT_MAP):
        return None

    return mapping_list


# ---------- Main driver ----------
def main():
    p = argparse.ArgumentParser(description="Verify colour-preserving automorphisms of a coloured Cayley graph")
    p.add_argument("--zN", type=int, default=24, help="size of cyclic Z_n factor (default 24)")
    p.add_argument("--k3", type=int, default=3, help="size of Z_k factor (default 3)")
    p.add_argument("--s1", type=int, default=1, help="index in canonical S3 list to use for s1 (0-5)")
    p.add_argument("--s2", type=int, default=3, help="index in canonical S3 list to use for s2 (0-5)")
    p.add_argument("--s1_perm", type=parse_perm, default=None, help="explicit permutation for s1 as comma list, e.g. '1,0,2'")
    p.add_argument("--s2_perm", type=parse_perm, default=None, help="explicit permutation for s2 as comma list")
    p.add_argument("--outdir", type=str, default="./verify_out", help="output directory")
    p.add_argument("--save_extra", action="store_true", help="save extra automorphisms json if found")
    args = p.parse_args()

    zn = args.zN
    k3 = args.k3
    if zn <= 0:
        raise SystemExit("zN must be positive")
    if k3 <= 0:
        raise SystemExit("k3 must be positive")

    # select permutations for s1 and s2
    if args.s1_perm is not None:
        s1p = args.s1_perm
    else:
        if 0 <= args.s1 < len(S3_all):
            s1p = S3_all[args.s1]
        else:
            raise SystemExit(f"s1 index out of range (0..{len(S3_all)-1}): {args.s1}")

    if args.s2_perm is not None:
        s2p = args.s2_perm
    else:
        if 0 <= args.s2 < len(S3_all):
            s2p = S3_all[args.s2]
        else:
            raise SystemExit(f"s2 index out of range (0..{len(S3_all)-1}): {args.s2}")

    # define generators (gen_name, element, colour)
    gens = [
        ('a', (1, (0, 1, 2), 0), 'red'),      # generator of Z_n
        ('s1', (0, s1p, 0), 'blue'),          # S3 generator s1
        ('s2', (0, s2p, 0), 'green'),         # S3 generator s2
        ('c', (0, (0, 1, 2), 1), 'yellow'),   # generator of Z_k
    ]

    os.makedirs(args.outdir, exist_ok=True)
    GROUP, GROUP_INDEX, ADJ, OUT_MAP = build_group_and_index(zn, k3, gens)
    N = len(GROUP)
    ID = group_identity(zn)
    ID_IDX = GROUP_INDEX[ID]

    # 1) check left multiplications
    left_failures = []
    for h in GROUP:
        mapping = left_mul_map(h, GROUP, GROUP_INDEX, zn, k3)
        if not check_colour_preserving(mapping, ADJ, OUT_MAP):
            left_failures.append(h)

    # 2) search for colour-preserving automorphisms by image-of-identity propagation
    found_mappings = {}
    for target_idx in range(N):
        mapping = build_mapping_from_image_of_identity(target_idx, ID_IDX, ADJ, OUT_MAP)
        if mapping is not None:
            found_mappings[target_idx] = mapping

    # compare to left multiplications
    left_maps = {tuple(left_mul_map(h, GROUP, GROUP_INDEX, zn, k3)): h for h in GROUP}
    extra = []
    for tgt, mapping in found_mappings.items():
        key = tuple(mapping)
        if key not in left_maps:
            extra.append((tgt, mapping))

    # write report
    report_path = os.path.join(args.outdir, "verification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Cayley graph colour-preserving automorphism verification\n")
        f.write("-------------------------------------------------------\n")
        f.write(f"group parameters: Z_{zn} × S3 × Z_{k3} (implemented as direct product)\n")
        f.write("generators used:\n")
        for name, elem, col in gens:
            f.write(f"  {name}: {elem} color={col}\n")
        f.write(f"group size: {N}\n")
        f.write(f"left multiplication colour-preserving check: {len(left_failures) == 0}\n")
        if left_failures:
            f.write("Left multiplication failures (first few):\n")
            for h in left_failures[:10]:
                f.write(f"  {h}\n")
        f.write(f"\ncolour-preserving mappings found by propagation (images of identity): {len(found_mappings)}\n")
        f.write(f"number of extra (non-left) mappings found: {len(extra)}\n")
        if extra:
            f.write("Extra mappings (target index list):\n")
            for tgt, mapping in extra:
                f.write(f"  target_idx={tgt}, target_element={GROUP[tgt]}\n")
            if args.save_extra:
                f.write("\nSee extra_automorphisms.json for mapping details\n")
            else:
                f.write("\nRun with --save_extra to save mapping details to JSON\n")
        else:
            f.write("No extra colour-preserving mappings discovered: all colour-preserving automorphisms correspond to left multiplications.\n")

    print("Wrote report to", report_path)

    # optionally write extra automorphisms
    if extra and args.save_extra:
        export = []
        for tgt, mapping in extra:
            export.append({"target_idx": int(tgt), "mapping": mapping, "target_element": GROUP[tgt]})
        extra_path = os.path.join(args.outdir, "extra_automorphisms.json")
        with open(extra_path, "w", encoding="utf-8") as ef:
            json.dump(export, ef, indent=2, ensure_ascii=False)
        print("Wrote extra automorphisms to", extra_path)


if __name__ == "__main__":
    main()
