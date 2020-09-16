"""
"""
import os
from time import time
from copy import deepcopy
from itertools import repeat
from typing import Union, Optional, List, Tuple, Set, Sequence, NoReturn, Any
from numbers import Real, Number

import numpy as np
import pandas as pd
import networkx as nx
from networkx import algorithms as NXA
import nxmetis
from easydict import EasyDict as ED

from .bipartite_graph import BipartiteGraph


__all__ = [
    "RBBDF", "RBBDF_v2",
]


def RBBDF(df_or_arr:Union[pd.DataFrame, np.ndarray, Sequence], density_threshold:float, max_iter:int=300, tol_increment:Optional[float]=None, tol_time:Optional[float]=None, fmt:str="rbbdf", verbose:int=0) -> Tuple[Union[Sequence[pd.DataFrame], pd.DataFrame], dict]:
    """
    """
    start = time()
    if isinstance(df_or_arr, pd.DataFrame):
        bg = BipartiteGraph.from_dataframe(df_or_arr)
    elif isinstance(df_or_arr, (np.ndarray, Sequence)):
        bg = BipartiteGraph.from_array(df_or_arr)

    # init
    diags, is_border = [list(bg.nodes())], [False]
    prev_density = 0
    density = bg.density

    if verbose >= 1:
        print(f"init density = {density}")
    
    # recursive iteration
    n_iter = 0
    while prev_density < density < density_threshold:
        if verbose >= 1:
            print("*"*110)
            print(f"in the {n_iter}-th iteration...")
        prev_density = density
        sorted_inds = np.argsort([bg.subgraph(item).size for item in diags])[::-1]
        true_idx = 0
        for idx in sorted_inds:
            if is_border[idx]:
                continue
            true_idx += 1
            B = bg.subgraph(diags[idx])
            sep_nodes, part1_nodes, part2_nodes = nxmetis.vertex_separator(B)
            skip_cond = \
                (bg.subgraph(part1_nodes).size==0) \
                or (bg.subgraph(part2_nodes).size==0)
            if skip_cond:
                continue
            if bg.subgraph(part1_nodes).size < bg.subgraph(part2_nodes).size:
                part1_nodes, part2_nodes = part2_nodes, part1_nodes
            # potential_diags = diags[:idx] + [part1_nodes, part2_nodes, sep_nodes] + diags[idx+1:]
            # potential_is_border = is_border[:idx] + [False,False,True] + is_border[idx+1:]
            potential_diags = diags[:idx] + [part1_nodes, part2_nodes]
            potential_is_border = is_border[:idx] + [False,False]
            if len(sep_nodes) > 0:
                potential_diags.append(sep_nodes)
                potential_is_border.append(True)
            potential_diags = potential_diags + diags[idx+1:]
            potential_is_border = potential_is_border + is_border[idx+1:]
            potential_density = \
                _compute_density(bg, potential_diags, potential_is_border)
            if potential_density > prev_density:
                density = potential_density
                diags = deepcopy(potential_diags)
                is_border = deepcopy(potential_is_border)
                if verbose >= 2:
                    print(f"at the {true_idx}-th largest block, density is improved from {prev_density} to {potential_density}")
                break
        if verbose >= 1:
            print(f"updated density = {density}, with prev_density = {prev_density}")
            if density >= density_threshold:
                print("density requirement is fulfilled!")
        if density - prev_density < (tol_increment or 0.0001*density_threshold):
            break
        if time() - start > (tol_time or 20*60):
            break
        n_iter += 1
        if n_iter > max_iter:
            break

    metadata = ED(
        diags=diags,
        is_border=is_border,
    )

    if fmt.lower() == "rbbdf":
        rows, cols = [], []
        for d in diags:
            newB = bg.subgraph(d)
            rows += newB.row_nodes
            cols += newB.col_nodes
        df = bg.to_dataframe(rows=rows,cols=cols)
    elif fmt.lower() == "bdf":
        X_tilde = _to_tilde_form(bg, diags, borders)
        df = [
            item.to_dataframe() for item in X_tilde
        ]

    if verbose == 0:
        metadata = ED()

    return df, metadata


def RBBDF_v2(df_or_arr:Union[pd.DataFrame, np.ndarray, Sequence], density_threshold:float, max_iter:int=300, tol_increment:Optional[float]=None, tol_time:Optional[float]=None, fmt:str="rbbdf", verbose:int=0) -> Tuple[Union[Sequence[pd.DataFrame], pd.DataFrame], dict]:
    """
    """
    start = time()
    if isinstance(df_or_arr, pd.DataFrame):
        bg = BipartiteGraph.from_dataframe(df_or_arr)
    elif isinstance(df_or_arr, (np.ndarray, Sequence)):
        bg = BipartiteGraph.from_array(df_or_arr)

    # init
    diags, is_border = [list(bg.nodes())], [False]
    prev_density = 0
    density = bg.density

    if verbose >= 1:
        print(f"init density = {density}")
    
    # recursive iteration
    n_iter = 0
    while prev_density < density < density_threshold:
        if verbose >= 1:
            print("*"*110)
            print(f"in the {n_iter}-th iteration...")
        prev_density = density
        sorted_inds = np.argsort([bg.subgraph(item).size for item in diags])[::-1]
        true_idx = 0
        for idx in sorted_inds:
            if is_border[idx]:
                continue
            true_idx += 1
            B = bg.subgraph(diags[idx])
            sep_nodes, part1_nodes, part2_nodes = nxmetis.vertex_separator(B)
            skip_cond = \
                (bg.subgraph(part1_nodes).size==0) \
                or (bg.subgraph(part2_nodes).size==0)
            if skip_cond:
                continue
            if bg.subgraph(part1_nodes).size < bg.subgraph(part2_nodes).size:
                part1_nodes, part2_nodes = part2_nodes, part1_nodes
            sb1 = bg.subgraph(part1_nodes).sorted_connected_components
            sb2 = bg.subgraph(part2_nodes).sorted_connected_components
            # potential_diags = diags[:idx] + [list(item.nodes) for item in sb1] + [list(item.nodes) for item in sb2] + [sep_nodes] + diags[idx+1:]
            # potential_is_border = is_border[:idx] + list(repeat(False, len(sb1)+len(sb2))) + [True] + is_border[idx+1:]
            potential_diags = diags[:idx] + [list(item.nodes) for item in sb1] + [list(item.nodes) for item in sb2]
            potential_is_border = is_border[:idx] + list(repeat(False, len(sb1)+len(sb2)))
            if len(sep_nodes) > 0:
                potential_diags.append(sep_nodes)
                potential_is_border.append(True)
            potential_diags = potential_diags + diags[idx+1:]
            potential_is_border = potential_is_border + is_border[idx+1:]
            potential_density = \
                _compute_density(bg, potential_diags, potential_is_border)
            if potential_density > prev_density:
                density = potential_density
                diags = deepcopy(potential_diags)
                is_border = deepcopy(potential_is_border)
                if verbose >= 2:
                    print(f"at the {true_idx}-th largest block, density is improved from {prev_density} to {potential_density}")
                break
        if verbose >= 1:
            print(f"updated density = {density}, with prev_density = {prev_density}")
            if density >= density_threshold:
                print("density requirement is fulfilled!")
        if density - prev_density < (tol_increment or 0.0001*density_threshold):
            break
        if time() - start > (tol_time or 20*60):
            break
        n_iter += 1
        if n_iter > max_iter:
            break

    metadata = ED(
        diags=diags,
        is_border=is_border,
    )

    if fmt.lower() == "rbbdf":
        rows, cols = [], []
        for d in diags:
            newB = bg.subgraph(d)
            rows += newB.row_nodes
            cols += newB.col_nodes
        df = bg.to_dataframe(rows=rows,cols=cols)
    elif fmt.lower() == "bdf":
        X_tilde = _to_tilde_form(bg, diags, borders)
        df = [
            item.to_dataframe() for item in X_tilde
        ]

    if verbose == 0:
        metadata = ED()

    return df, metadata


def _compute_density(bg:BipartiteGraph, diags:Sequence[Sequence[str]], is_border:Sequence[bool]) -> float:
    """
    """
    X_tilde = _to_tilde_form(bg, diags, is_border)
    density = sum([item.n_nonzeros for item in X_tilde]) / sum([item.size for item in X_tilde])
    return density


def _to_tilde_form(bg:BipartiteGraph, diags:Sequence[Sequence[str]], is_border:Sequence[bool]) -> Sequence[BipartiteGraph]:
    """
    """
    X_tilde = []
    for idx, (d, ib) in enumerate(zip(diags, is_border)):
        if ib:
            continue
        block_nodes = deepcopy(d)
        for new_idx in range(idx+1, len(diags)):
            if is_border[new_idx]:
                block_nodes += diags[new_idx]
        X_tilde.append(bg.subgraph(block_nodes))
    return X_tilde
