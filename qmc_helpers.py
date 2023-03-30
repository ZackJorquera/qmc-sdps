import numpy as np
import itertools


def gell_mann_mat(d, type, a, b=None):
    assert type in ['+', '-', 'd']
    assert 1 <= a <= d-1
    if type == 'd':
        assert b is None
    else:
        assert b is not None
        assert a < b <= d

    Gamma = np.zeros((d, d), dtype=np.complex_)

    if type == '+':
        Gamma[a-1, b-1] = 1
        Gamma[b-1, a-1] = 1
    elif type == '-':
        Gamma[a-1, b-1] = -1j
        Gamma[b-1, a-1] = 1j
    else:
        for b in range(a):
            Gamma[b,b] = 1
        Gamma[a, a] = -a
        Gamma *= np.sqrt(2/(a*(a+1)))
    return Gamma


def create_QMC_graph_ham(n, edges):
    # TODO: use sparse matrices
    H_G = np.zeros((2**n, 2**n), dtype=np.complex_)
    I = np.eye(2, dtype=np.complex_)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex_)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex_)
    Y = 1j * np.matmul(X, Z)

    for (i, j, w) in set(edges):
        assert(i != j)
        assert(i < n and j < n)
        if i > j: i, j = j, i
        h_ij = np.eye(2 ** n, dtype=np.complex_)

        for P in [X, Y, Z]:
            PtpP = np.eye(1, dtype=np.complex_)

            for k in range(0, i):
                PtpP = np.kron(PtpP, I)
            PtpP = np.kron(PtpP, P)
            for k in range(i+1, j):
                PtpP = np.kron(PtpP, I)
            PtpP = np.kron(PtpP, P)
            for k in range(j+1, n):
                PtpP = np.kron(PtpP, I)

            h_ij -= PtpP
        h_ij *= 1/4

        H_G += w * h_ij
    return H_G


def create_QMC_d_graph_ham(n, d, edges):
    # TODO: use sparse matrices
    H_G = np.zeros((d**n, d**n), dtype=np.complex_)
    I = np.eye(d, dtype=np.complex_)
    gell_mann_mats = [gell_mann_mat(d, '+', pair[0]+1, pair[1]+1) for pair in itertools.combinations(range(d), 2)] + \
                     [gell_mann_mat(d, '-', pair[0] + 1, pair[1] + 1) for pair in itertools.combinations(range(d), 2)] + \
                     [gell_mann_mat(d, 'd', a + 1) for a in range(d-1)]

    for (i, j, w) in set(edges):
        assert(i != j)
        assert(i < n and j < n)
        if i > j: i, j = j, i
        h_ij = np.eye(d ** n, dtype=np.complex_)
        h_ij *= (d-1)/d

        for P in gell_mann_mats:
            PtpP = np.eye(1, dtype=np.complex_)

            for k in range(0, i):
                PtpP = np.kron(PtpP, I)
            PtpP = np.kron(PtpP, P)
            for k in range(i+1, j):
                PtpP = np.kron(PtpP, I)
            PtpP = np.kron(PtpP, P)
            for k in range(j+1, n):
                PtpP = np.kron(PtpP, I)

            h_ij -= 1/2 * PtpP
        h_ij *= 1/2

        H_G += w * h_ij
    return H_G


if __name__ == "__main__":
    from graph_helpers import gen_cycle, gen_complete, gen_star
    d = 3
    n = d + 3

    star_graph = gen_star(n)[0]

    # QMC
    H_G = create_QMC_graph_ham(n, star_graph)

    w, v = np.linalg.eigh(H_G)
    max_energy1 = w[-1]
    x1 = np.round(v[:, -1], 10)

    # QMC_d
    Hd_G = create_QMC_d_graph_ham(n, d, star_graph)

    w, v = np.linalg.eigh(Hd_G)
    max_energy2 = w[-1]
    x2 = np.round(v[:, -1], 10)

    print(f"Max Energy for Star Graph ({d=}, {n=}), QMC: {max_energy1}, QMC_d: {max_energy2}")

    d = 4
    comp_graph = gen_complete(d)

    # QMC
    H_G = create_QMC_graph_ham(d, comp_graph)

    w, v = np.linalg.eigh(H_G)
    max_energy1 = w[-1]
    x1 = np.round(v[:, -1], 10)

    # QMC_d
    Hd_G = create_QMC_d_graph_ham(d, d, comp_graph)

    w, v = np.linalg.eigh(Hd_G)
    max_energy2 = w[-1]
    x2 = np.round(v[:, -1], 10)

    print(f"Max Energy for d-clique Graph ({d=}), QMC: {max_energy1}, QMC_d: {max_energy2}")
