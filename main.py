import numpy as np
import cvxpy as cp
import scipy
import itertools

from qmc_helpers import *


def goemans_williamson_max_cut(n, edges, ave_over_iters=None):
    # SDP for Max-Cut relaxation
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]  # X is PSD
    constraints += [
        X[i, i] == 1 for i in range(n)  # unit len
    ]
    objective = 0.5 * sum(w * (1 - X[i, j]) for (i, j, w) in edges)

    sdp = cp.Problem(cp.Maximize(objective), constraints)
    sdp.solve()
    sdp_val = sdp.value

    # GW Rounding
    x = np.real(scipy.linalg.sqrtm(X.value))
    if ave_over_iters is None:
        ave_over_iters = 1

    round_vals = np.zeros(ave_over_iters)

    for i in range(ave_over_iters):
        r = np.random.randn(n)
        round_x = np.sign(np.dot(x, r))
        round_vals[i] = 0.5 * sum(w * (1 - round_x[i] * round_x[j]) for (i, j, w) in edges)

    return sdp_val, np.average(round_vals)


def gharibian_parekh_rounding(vec_mat, edges, ave_over_iters=None):
    if ave_over_iters is None:
        ave_over_iters = 1

    round_vals = np.zeros(ave_over_iters)

    for i in range(ave_over_iters):
        Z = np.random.randn(3, vec_mat.shape[0])
        round_x_bloch_vecs = np.matmul(Z, vec_mat)
        round_x_bloch_vecs /= np.linalg.norm(round_x_bloch_vecs, axis=0)
        round_vals[i] = 0.25 * sum(
            w * (1 - np.dot(round_x_bloch_vecs[:, i], round_x_bloch_vecs[:, j])) for (i, j, w) in edges)

    return np.average(round_vals)


def product_state_sdp_q_max_cut(n, edges, ave_over_iters=None, real=True):
    # SDP for Q-Max-Cut relaxation
    # let P_i be the index 3 * i + ind(P), where i in [n] and p = ind(P) = 0, 1, 2 for P = X, Y, Z
    sym_herm = {'symmetric': True} if real else {'hermitian': True}

    # Note, we ignore indexing the identity because it isn't needed.
    X = cp.Variable((3 * n, 3 * n), **sym_herm)
    constraints = [X >> 0]  # PSD
    constraints += [
        X[i, i] == 1 for i in range(3 * n)  # Unit Len
    ]
    if real:  # real symmetric case
        # Commuting Paulis constraints are already baked into the fact that X is real sym
        constraints += [
            # Anti-commuting Paulis
            X[3 * i + p[0], 3 * i + p[1]] == 0
            for i in range(n) for p in itertools.combinations(range(3), 2)
        ]
    else:
        # hermitian case
        constraints += [
            # Commuting Paulis
            X[3 * pair[0] + p[0], 3 * pair[1] + p[1]] == X[3 * pair[1] + p[1], 3 * pair[0] + p[0]]  # only allow real values for these entries
            for pair in itertools.combinations(range(n), 2)
            for p in itertools.product(range(3), repeat=2)
        ]
        constraints += [
            # Anti-commuting Paulis
            X[3 * i + p[0], 3 * i + p[1]] == -X[3 * i + p[1], 3 * i + p[0]]  # only imaginary values for these entries
            for i in range(n) for p in itertools.combinations(range(3), 2)
        ]

    objective = 0.25 * sum(
        w * (1 - X[3 * i + 0, 3 * j + 0] - X[3 * i + 1, 3 * j + 1] - X[3 * i + 2, 3 * j + 2]) for (i, j, w) in edges)
    if not real:
        objective = cp.real(objective)  # Note, by the 'Commuting Paulis' constraint they are already real

    sdp = cp.Problem(cp.Maximize(objective), constraints)
    sdp.solve()
    sdp_val = sdp.value

    # GP Rounding
    x = np.real(scipy.linalg.sqrtm(X.value))
    x = np.reshape(x, (9 * n, n), order='F') / np.sqrt(3)

    round_val_ave = gharibian_parekh_rounding(x, edges, ave_over_iters)

    return sdp_val, round_val_ave


def product_state_sdp_q_max_cut_simplified(n, edges, ave_over_iters=None):
    # SDP for Q-Max-Cut relaxation, from Definition 2.13 in John Wright paper
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]  # X is PSD
    constraints += [
        X[i, i] == 1 for i in range(n)  # unit len
    ]
    objective = 0.25 * sum(w * (1 - 3 * X[i, j]) for (i, j, w) in edges)

    sdp = cp.Problem(cp.Maximize(objective), constraints)
    sdp.solve()
    sdp_val = sdp.value

    # GW Rounding
    x = np.real(scipy.linalg.sqrtm(X.value))

    round_val_ave = gharibian_parekh_rounding(x, edges, ave_over_iters)

    return sdp_val, round_val_ave


def lasserre_2_product_state_q_max_cut_relaxed(n, edges, ave_over_iters=None, obj_type=2):
    # This is the relaxed Lasserre 2 SDP for QMC from Def 5 in the Eunou Lee paper

    # let P_i be the index 3 * i + ind(P) + 1, where i \in [n] and p = ind(P) = 0, 1, 2 for P = X, Y, Z
    ind1 = lambda p, i: 1 + 3 * i + p

    # Then for degree 2 terms, let P_i P_j for i < j be indexed with the following
    # offset + 3*(j*(j-1)//2 + i) + ind(P), where offset is the nubmer of degree 1 indices (3 * n + 1)
    def ind2(p, i, j):
        assert i != j
        assert p in [0, 1, 2]
        offset = 3 * n + 1
        return offset + 3 * (j * (j - 1) // 2 + i) + p if i < j else ind2(p, j, i)

    mat_size = (3*n + 1) + 3 * (n * (n-1) // 2)
    X = cp.Variable((mat_size, mat_size), symmetric=True)

    constraints = [X >> 0]  # PSD
    constraints += [
        X[i, i] == 1 for i in range(mat_size)  # Unit Len
    ]
    # Commuting Paulis constraints are already baked into the fact that X is real sym
    constraints += [
        # Anti-commuting Paulis
        X[ind1(p[0], i), ind1(p[1], i)] == 0
        for i in range(n) for p in itertools.combinations(range(3), 2)
    ]

    # degree 2 constraints
    constraints += [
        # M(P_i, P_j) = M(P_i P_j, I), Identity
        X[ind2(p, pair[0], pair[1]), 0] == X[ind1(p, pair[0]), ind1(p, pair[1])]
        for pair in itertools.combinations(range(n), n) for p in range(3)
    ]

    # For any 2 pairs of paulis with one overlapping and the paulis the same
    constraints += [
        # M(P_i P_j, P_j P_k) = M(P_i P_j, I)
        X[ind2(p, tri[pair[0]], tri[pair[1]]), ind2(p, tri[third], tri[pair[dup[0]]])]
            == X[ind2(p, tri[pair[dup[1]]], tri[third]), 0]
        for tri in itertools.combinations(range(n), 3) for pair in itertools.combinations(range(3), 2)
        for dup in [(0, 1), (1, 0)] for third in set(range(3)).difference(pair)
        for p in range(3)
    ]

    # For any 2 pairs of paulis with one overlapping and paulis are different
    constraints += [
        # M(P_i P_j, Q_j Q_k) = 0
        X[ind2(p[0], tri[pair[0]], tri[pair[1]]), ind2(p[1], tri[third], tri[pair[dup[0]]])] == 0
        for tri in itertools.combinations(range(n), 3) for pair in itertools.combinations(range(3), 2)
        for dup in [(0, 1), (1, 0)] for third in set(range(3)).difference(pair)
        for p in itertools.combinations(range(3), 2)
    ]

    # Two overlapping pairs of paulis that are different
    constraints += [
        # M(P_i P_j, Q_i Q_j) = -M(P'_i P'_j, I)
        X[ind2(p[0], pair[0], pair[1]), ind2(p[1], pair[0], pair[1])] == -X[ind2(p[2], pair[0], pair[1]), 0]
        for pair in itertools.combinations(range(n), 2) for p in itertools.permutations(range(3))
    ]

    objective = 0.25 * sum(
        w * (1 - X[ind2(0, i, j), 0] - X[ind2(1, i, j), 0] - X[ind2(2, i, j), 0]) for (i, j, w) in edges)

    sdp = cp.Problem(cp.Maximize(objective), constraints)
    sdp.solve()
    sdp_val = sdp.value

    # GP Rounding
    x = np.real(scipy.linalg.sqrtm(X.value))
    x = np.reshape(x[:, 1:3*n+1], (3 * mat_size, n), order='F') / np.sqrt(3)

    round_val_ave = gharibian_parekh_rounding(x, edges, ave_over_iters)

    return sdp_val, round_val_ave


def gen_cycle(n):
    edges = [(i, (i + 1) % n, 1) for i in range(n)]
    max_cut = n - (n % 2)

    H = create_QMC_graph_ham(n, edges)
    q_max_cut = np.max(np.linalg.eigvalsh(H))

    return edges, max_cut, q_max_cut


def gen_star(n, calc_q_max_cut=False):
    edges = [(0, i + 1, 1) for i in range(n-1)]
    max_cut = n-1
    q_max_cut = n/2
    if calc_q_max_cut:
        H = create_QMC_graph_ham(n, edges)
        q_max_cut = np.max(np.linalg.eigvalsh(H))
    return edges, max_cut, q_max_cut


def star_test():
    # we test the star graphs using known values
    for n in range(2, 9):
        # we expect opt_qmc_val to be n/2 (as per [AGM20])
        edges, opt_mc_val, opt_qmc_val = gen_star(n, False)
        assert np.allclose(opt_qmc_val, n/2)

        # we expect the sdp_val to be n-1 (as per [PT21a])
        gp_sdp_val, _ = product_state_sdp_q_max_cut_simplified(n, edges)
        # gp_sdp_val, _ = product_state_sdp_q_max_cut(n, edges)
        assert np.allclose(gp_sdp_val, n-1, atol=10e-4)

        # For Lasserre 2, we expect the las_sdp_val to be n/2 (as per [PT21a])
        las_sdp_val, _ = lasserre_2_product_state_q_max_cut_relaxed(n, edges)
        assert np.allclose(las_sdp_val, n/2, atol=10e-4)


def calc_gaps_of_graph_las1(n, edges):
    H_G = create_QMC_graph_ham(n, edges)
    opt_qmc_val = np.max(np.linalg.eigvalsh(H_G))

    # gp_sdp_val, gp_round_val = product_state_sdp_q_max_cut(n, edges, 100)
    gp_sdp_val, gp_round_val = product_state_sdp_q_max_cut_simplified(n, edges, 100)

    print("Gharibian-Parekh for QMC:")
    print(f"{gp_sdp_val=:.4f}, {gp_round_val=:.4f}, {opt_qmc_val=:.4f}")
    print(f"Ratio: {gp_round_val/gp_sdp_val=:.4f}, Algorithmic gap: {gp_round_val/opt_qmc_val=:.4f}, Integrality gap: {opt_qmc_val/gp_sdp_val=:.4f}")


def calc_gaps_of_graph_las2(n, edges):
    H_G = create_QMC_graph_ham(n, edges)
    opt_qmc_val = np.max(np.linalg.eigvalsh(H_G))

    las2_sdp_val, las2_round_val = lasserre_2_product_state_q_max_cut_relaxed(n, edges, 100)

    print("Las 2 Gharibian-Parekh for QMC:")
    print(f"{las2_sdp_val=:.4f}, {las2_round_val=:.4f}, {opt_qmc_val=:.4f}")
    print(f"Ratio: {las2_round_val/las2_sdp_val=:.4f}, Algorithmic gap: {las2_round_val/opt_qmc_val=:.4f}, Integrality gap: {opt_qmc_val/las2_sdp_val=:.4f}")


if __name__ == "__main__":
    # star_test()
    n = 9
    edges, opt_mc_val, opt_qmc_val = gen_cycle(n)
    sdp_val, round_val = goemans_williamson_max_cut(n, edges, 100)

    print(f"Cycle of len {n}:")
    print(f"Goemans-Williamson for Max-Cut:")
    print(f"{sdp_val=:.4f}, {round_val=:.4f}, {opt_mc_val=:.4f}")
    print(f"Ratio: {round_val/sdp_val=:.4f}, Algorithmic gap: {round_val/opt_mc_val=:.4f}, Integrality gap: {opt_mc_val/sdp_val=:.4f}")  # note, GW ratio is \approx 0.8785

    print("")
    calc_gaps_of_graph_las1(n, edges)

    print("")
    calc_gaps_of_graph_las2(n, edges)

