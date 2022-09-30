"""
References:
-----------
[1] Zhang, Yongfeng, et al. "Localized matrix factorization for recommendation based on matrix block diagonal forms." Proceedings of the 22nd international conference on World Wide Web. 2013.

key method: `nxmetis.vertex_separator`

TODO:
consider GPVS with weighted edges
"""

from time import time
from copy import deepcopy
from itertools import repeat
from typing import Union, Optional, Tuple, Sequence
from numbers import Real

import numpy as np
import pandas as pd
import nxmetis
from easydict import EasyDict as ED
from tqdm.auto import tqdm

from .bipartite_graph import BipartiteGraph, ScipySparseMatrix


__all__ = [
    "RBBDF",
    "RBBDF_v2",
]


def RBBDF(
    df_or_arr: Union[pd.DataFrame, np.ndarray, ScipySparseMatrix, Sequence],
    density_threshold: float,
    vs_options: Optional[nxmetis.MetisOptions] = None,
    max_iter: int = 200,
    tol_increment: Optional[float] = None,
    tol_time: Optional[Real] = None,
    fmt: str = "rbbdf",
    verbose: int = 1,
) -> Tuple[Union[Sequence[pd.DataFrame], pd.DataFrame], dict]:
    """finished, checked,

    transform a matrix-like into its RBBDF (Recursive Bordered Block Diagonal Form)
    or its corresponding BDF (Block Diagonal Form) recursively

    Parameters:
    -----------
    df_or_arr: DataFrame or array_like,
        the DataFrame or array to transform into RBBDF
    density_threshold: float,
        termination condition of `BipartiteGraph` density
    vs_options: MetisOptions, optional,
        options for `nxmetis.vertex_separator`
    max_iter: int, default 200,
        termination condition of number of iterations
    tol_increment: float, optional,
        termination condition of increment of density of each iteration
        if is None, defaults to 0.0001 * `density_threshold`
    tol_time: real number, optional,
        termination condition of duration of iteration
        if is None, defaults to 20 minutes
    fmt: str, default "rbbdf",
        format of the output, can also be "bdf"
    verbose: int, default 1,
        print verbosity

    Returns:
    --------
    df: DataFrame or sequence of DataFrame,
        `df_or_arr` in RBBDF or in BDF
    metadata: dict,
        diags: sequence of sequence of str,
            each sub-sequence is a sequence of names of the rows and columns of a diagonal block
        is_border: sequence of bool,
            "border" indicators for elements in `diags`

    """
    assert fmt.lower() in ["rbbdf", "bdf"]
    start = time()
    if isinstance(df_or_arr, pd.DataFrame):
        df = df_or_arr.copy()
        df.index = df.index.map(str)
        df.columns = df.columns.map(str)
        bg = BipartiteGraph.from_dataframe(df)
    elif isinstance(df_or_arr, ScipySparseMatrix.__args__):
        bg = BipartiteGraph.from_sparse(df_or_arr)
    elif isinstance(df_or_arr, (np.ndarray, Sequence)):
        bg = BipartiteGraph.from_array(df_or_arr)
    else:
        raise TypeError(f"unsupported type of `df_or_arr`: {type(df_or_arr)}")

    # init
    diags, is_border = [list(bg.nodes())], [False]
    prev_density = 0
    density = bg.density

    if verbose >= 1:
        print(f"init density = {density}")

    # recursive iteration
    with tqdm(
        range(max_iter), disable=not verbose, total=max_iter, desc="Iter"
    ) as pbar:
        for n_iter in pbar:
            postfix_str = ""
            prev_density = density
            sorted_inds = np.argsort([bg.subgraph(item).size for item in diags])[::-1]
            true_idx = 0
            for idx in sorted_inds:
                if is_border[idx]:
                    continue
                true_idx += 1
                B = bg.subgraph(diags[idx])
                sep_nodes, part1_nodes, part2_nodes = nxmetis.vertex_separator(
                    B, options=vs_options
                )
                skip_cond = (bg.subgraph(part1_nodes).size == 0) or (
                    bg.subgraph(part2_nodes).size == 0
                )
                if skip_cond:
                    continue
                if bg.subgraph(part1_nodes).size < bg.subgraph(part2_nodes).size:
                    part1_nodes, part2_nodes = part2_nodes, part1_nodes
                # potential_diags = diags[:idx] + [part1_nodes, part2_nodes, sep_nodes] + diags[idx+1:]
                # potential_is_border = is_border[:idx] + [False,False,True] + is_border[idx+1:]
                potential_diags = diags[:idx] + [part1_nodes, part2_nodes]
                potential_is_border = is_border[:idx] + [False, False]
                if len(sep_nodes) > 0:
                    potential_diags.append(sep_nodes)
                    potential_is_border.append(True)
                potential_diags = potential_diags + diags[idx + 1 :]
                potential_is_border = potential_is_border + is_border[idx + 1 :]
                potential_density = _compute_density(
                    bg, potential_diags, potential_is_border
                )
                if potential_density > prev_density:
                    density = potential_density
                    diags = deepcopy(potential_diags)
                    is_border = deepcopy(potential_is_border)
                    if verbose >= 2:
                        postfix_str += f"at the {true_idx}-th largest block, density {prev_density:.5f} -> {potential_density:.5f}."
                    break
            if verbose >= 1:
                postfix_str += f" density {prev_density:.5f} -> {density:.5f}"
                if density >= density_threshold:
                    postfix_str += ", density requirement is fulfilled!"
                pbar.set_postfix_str(postfix_str)

            # termination conditions
            if density >= density_threshold:
                if verbose >= 1:
                    print("density requirement is fulfilled!")
                break
            if density - prev_density < (tol_increment or 0.0001 * density_threshold):
                if verbose >= 1:
                    print("density increment is too small!")
                break
            if time() - start > (tol_time or 20 * 60):
                if verbose >= 1:
                    print("time limit is reached!")
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
        df = bg.to_dataframe(rows=rows, cols=cols)
    elif fmt.lower() == "bdf":
        X_tilde = _to_tilde_form(bg, diags, is_border)
        df = [item.to_dataframe() for item in X_tilde]

    if verbose == 0:
        metadata = ED()

    return df, metadata


def RBBDF_v2(
    df_or_arr: Union[pd.DataFrame, np.ndarray, ScipySparseMatrix, Sequence],
    density_threshold: float,
    vs_options: Optional[nxmetis.MetisOptions] = None,
    max_iter: int = 200,
    tol_increment: Optional[float] = None,
    tol_time: Optional[Real] = None,
    fmt: str = "rbbdf",
    verbose: int = 0,
) -> Tuple[Union[Sequence[pd.DataFrame], pd.DataFrame], dict]:
    """finished, checked,

    transform a matrix-like into its RBBDF (Recursive Bordered Block Diagonal Form)
    or its corresponding BDF (Block Diagonal Form) recursively

    the difference compared to `RBBDF` is that
    after each bisection, each of the 2 non-border diagonal part
    will be further split into several diagonal blocks if applicable

    Parameters:
    -----------
    df_or_arr: DataFrame or array_like,
        the DataFrame or array to transform into RBBDF
    density_threshold: float,
        termination condition of `BipartiteGraph` density
    vs_options: MetisOptions, optional,
        options for `nxmetis.vertex_separator`
    max_iter: int, default 200,
        termination condition of number of iterations
    tol_increment: float, optional,
        termination condition of increment of density of each iteration
        if is None, defaults to 0.0001 * `density_threshold`
    tol_time: real number, optional,
        termination condition of duration of iteration
        if is None, defaults to 20 minutes
    fmt: str, default "rbbdf", case insensitive,
        format of the output, can also be "bdf"
    verbose: int, default 0,
        print verbosity

    Returns:
    --------
    df: DataFrame or sequence of DataFrame,
        `df_or_arr` in RBBDF or in BDF
    metadata: dict,
        diags: sequence of sequence of str,
            each sub-sequence is a sequence of names of the rows and columns of a diagonal block
        is_border: sequence of bool,
            "border" indicators for elements in `diags`

    """
    assert fmt.lower() in ["rbbdf", "bdf"]
    start = time()
    if isinstance(df_or_arr, pd.DataFrame):
        df = df_or_arr.copy()
        df.index = df.index.map(str)
        df.columns = df.columns.map(str)
        bg = BipartiteGraph.from_dataframe(df)
    elif isinstance(df_or_arr, ScipySparseMatrix.__args__):
        bg = BipartiteGraph.from_sparse(df_or_arr)
    elif isinstance(df_or_arr, (np.ndarray, Sequence)):
        bg = BipartiteGraph.from_array(df_or_arr)
    else:
        raise TypeError(f"unsupported type of `df_or_arr`: {type(df_or_arr)}")

    # init
    diags, is_border = [list(bg.nodes())], [False]
    prev_density = 0
    density = bg.density

    if verbose >= 1:
        print(f"init density = {density}")

    # recursive iteration
    with tqdm(
        range(max_iter), disable=not verbose, total=max_iter, desc="Iter"
    ) as pbar:
        for n_iter in pbar:
            postfix_str = ""
            prev_density = density
            sorted_inds = np.argsort([bg.subgraph(item).size for item in diags])[::-1]
            true_idx = 0
            for idx in sorted_inds:
                if is_border[idx]:
                    continue
                true_idx += 1
                B = bg.subgraph(diags[idx])
                sep_nodes, part1_nodes, part2_nodes = nxmetis.vertex_separator(
                    B, options=vs_options
                )
                skip_cond = (bg.subgraph(part1_nodes).size == 0) or (
                    bg.subgraph(part2_nodes).size == 0
                )
                if skip_cond:
                    continue
                if bg.subgraph(part1_nodes).size < bg.subgraph(part2_nodes).size:
                    part1_nodes, part2_nodes = part2_nodes, part1_nodes
                sb1 = bg.subgraph(part1_nodes).sorted_connected_components
                sb2 = bg.subgraph(part2_nodes).sorted_connected_components
                # potential_diags = diags[:idx] + [list(item.nodes) for item in sb1] + [list(item.nodes) for item in sb2] + [sep_nodes] + diags[idx+1:]
                # potential_is_border = is_border[:idx] + list(repeat(False, len(sb1)+len(sb2))) + [True] + is_border[idx+1:]
                potential_diags = (
                    diags[:idx]
                    + [list(item.nodes) for item in sb1]
                    + [list(item.nodes) for item in sb2]
                )
                potential_is_border = is_border[:idx] + list(
                    repeat(False, len(sb1) + len(sb2))
                )
                if len(sep_nodes) > 0:
                    potential_diags.append(sep_nodes)
                    potential_is_border.append(True)
                potential_diags = potential_diags + diags[idx + 1 :]
                potential_is_border = potential_is_border + is_border[idx + 1 :]
                potential_density = _compute_density(
                    bg, potential_diags, potential_is_border
                )
                if potential_density > prev_density:
                    density = potential_density
                    diags = deepcopy(potential_diags)
                    is_border = deepcopy(potential_is_border)
                    if verbose >= 2:
                        postfix_str += f"at the {true_idx}-th largest block, density {prev_density:.5f} -> {potential_density:.5f}."
                    break
            if verbose >= 1:
                postfix_str += f" density {prev_density:.5f} -> {density:.5f}"
                if density >= density_threshold:
                    postfix_str += ", density requirement is fulfilled!"
                pbar.set_postfix_str(postfix_str)

            # termination conditions
            if density >= density_threshold:
                if verbose >= 1:
                    print("density requirement is fulfilled!")
                break
            if density - prev_density < (tol_increment or 0.0001 * density_threshold):
                if verbose >= 1:
                    print("density increment is too small!")
                break
            if time() - start > (tol_time or 20 * 60):
                if verbose >= 1:
                    print("time limit is reached!")
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
        df = bg.to_dataframe(rows=rows, cols=cols)
    elif fmt.lower() == "bdf":
        X_tilde = _to_tilde_form(bg, diags, is_border)
        df = [item.to_dataframe() for item in X_tilde]

    if verbose == 0:
        metadata = ED()

    return df, metadata


def _compute_density(
    bg: BipartiteGraph, diags: Sequence[Sequence[str]], is_border: Sequence[bool]
) -> float:
    """finished, checked,

    compute density for a RBBDF BipartiteGraph

    Parameters:
    -----------
    bg: BipartiteGraph,
        the `BipartiteGraph` to compute density from whose corresponding matrix
    diags: sequence of sequence of str,
        each subsequence is a sequence of names of the nodes in `bg`,
        which corresponds to a diagonal block in the corresponding matrix
    is_border: sequence of bool,
        "border" indicators for elements in `diags`,

    Returns:
    --------
    rho: float,
        density of the `BipartiteGraph` `bg`

    """
    X_tilde = _to_tilde_form(bg, diags, is_border)
    rho = sum([item.n_nonzeros for item in X_tilde]) / sum(
        [item.size for item in X_tilde]
    )
    return rho


def _to_tilde_form(
    bg: BipartiteGraph, diags: Sequence[Sequence[str]], is_border: Sequence[bool]
) -> Sequence[BipartiteGraph]:
    """finished, checked,

    transforms a `BipartiteGraph` to the corresponding BDF

    Parameters:
    -----------
    bg: BipartiteGraph,
        the `BipartiteGraph` to compute density from whose corresponding matrix
    diags: sequence of sequence of str,
        each subsequence is a sequence of names of the nodes in `bg`,
        which corresponds to a diagonal block in the corresponding matrix
    is_border: sequence of bool,
        "border" indicators for elements in `diags`,

    Returns:
    --------
    X_tilde: sequence of BipartiteGraph,
        each element is constructed from `bg`, according to rules of ref. [1]

    """
    X_tilde = []
    for idx, (d, ib) in enumerate(zip(diags, is_border)):
        if ib:
            continue
        block_nodes = deepcopy(d)
        for new_idx in range(idx + 1, len(diags)):
            if is_border[new_idx]:
                block_nodes += diags[new_idx]
        X_tilde.append(bg.subgraph(block_nodes))
    return X_tilde


def _rbbdf_to_bdf(
    df_rbbdf: pd.DataFrame, diags: Sequence[Sequence[str]], is_border: Sequence[bool]
) -> Sequence[BipartiteGraph]:
    """finished, checked,

    transforms a RBBDF `DataFrame` of to the format of BDF

    Parameters:
    -----------
    df_rbbdf: DataFrame,
        the RBBDF `DataFrame` to transform to the format of BDF
    diags: sequence of sequence of str,
        each subsequence is a sequence of names of the nodes in `bg`,
        which corresponds to a diagonal block in the corresponding matrix
    is_border: sequence of bool,
        "border" indicators for elements in `diags`,

    Returns:
    --------
    X_tilde: sequence of BipartiteGraph,
        each element is constructed from `bg`, according to rules of ref. [1]

    """
    bg = BipartiteGraph.from_dataframe(df_rbbdf)
    X_tilde = _to_tilde_form(bg, diags, is_border)
    X_tilde = [bg.subgraph(item).to_dataframe() for item in X_tilde]
    return X_tilde
