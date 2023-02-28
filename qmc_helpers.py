import numpy as np


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