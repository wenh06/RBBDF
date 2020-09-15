"""

References:
-----------
[1] Zhang, Yongfeng, et al. "Localized matrix factorization for recommendation based on matrix block diagonal forms." Proceedings of the 22nd international conference on World Wide Web. 2013.
[2] Aykanat, Cevdet, Ali Pinar, and Ümit V. Çatalyürek. "Permuting sparse rectangular matrices into block-diagonal form." SIAM Journal on scientific computing 25.6 (2004): 1860-1879.
"""

from .bipartite_graph import BipartiteGraph
from .rbbdf import RBBDF, RBBDF_v2


__all__ = [
    "BipartiteGraph",
    "RBBDF", "RBBDF_v2",
]
