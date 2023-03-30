from qmc_helpers import *
import networkx as nx
import itertools


def gen_cycle(n, calc_evec=False):
    edges = [(i, (i + 1) % n, 1) for i in range(n)]
    max_cut = n - (n % 2)

    H = create_QMC_graph_ham(n, edges)
    if calc_evec:
        evals, evecs = np.linalg.eigh(H)
        q_max_cut = evals[-1]
        q_max_cut_soln = evecs[:, -1]/np.linalg.norm(evecs[:, -1])
    else:
        q_max_cut = -1  # np.max(np.linalg.eigvalsh(H))

    return edges, max_cut, q_max_cut


def gen_star(n, calc_q_max_cut=False, calc_evec=False):
    edges = [(0, i + 1, 1) for i in range(n-1)]
    max_cut = n-1
    q_max_cut = n/2
    if calc_q_max_cut:
        H = create_QMC_graph_ham(n, edges)
        if calc_evec:
            evals, evecs = np.linalg.eigh(H)
            q_max_cut = evals[-1]
            q_max_cut_soln = evecs[:, -1]
        else:
            q_max_cut = np.max(np.linalg.eigvalsh(H))
    return edges, max_cut, q_max_cut


def gen_rand(n, degree):
    graph = nx.random_regular_graph(degree, n)
    edges = [(i, j, 1) for (i, j) in graph.edges]

    max_cut = max(
        sum(w * (z[i] + z[j] - 2 * z[i] * z[j]) for i, j, w in edges) for z in itertools.product(range(2), repeat=n))

    H = create_QMC_graph_ham(n, edges)
    q_max_cut = np.max(np.linalg.eigvalsh(H))

    return edges, max_cut, q_max_cut


def gen_complete(n):
    edges = [(pair[0], pair[1], 1) for pair in itertools.combinations(range(n), 2)]
    return edges
