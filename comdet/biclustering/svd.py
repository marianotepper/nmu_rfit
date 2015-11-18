from __future__ import absolute_import
import numpy as np
import comdet.biclustering.utils as utils


class SVD:

    def __init__(self, array, rcond=1e-15):
        self.shape = array.shape
        self.u, self.s, self.vt = np.linalg.svd(array, full_matrices=False)
        self.rcond = rcond

    def update(self, a, b):
        m = self.u.T.dot(a)
        p = a - self.u.dot(m)
        r_a = utils.frobenius_norm(p)
        p /= r_a

        n = self.vt.dot(b)
        q = b - self.vt.T.dot(n)
        r_b = utils.frobenius_norm(q)
        q /= r_b

        u_a = np.append(m, [r_a])
        v_b = np.append(n, [r_b])

        k = np.diag(np.append(self.s, [0]))
        k += np.outer(u_a, v_b)

        inner_u, s_new, inner_vt = np.linalg.svd(k)

        self.u = np.dot(np.hstack((self.u, np.atleast_2d(p).T)), inner_u)
        self.s = s_new
        self.vt = np.dot(inner_vt, np.vstack((self.vt, np.atleast_2d(q))))

    def remove_column(self, idx):
        b = np.zeros((self.shape[1],))
        b[idx] = 1
        n = self.vt[:, idx]
        q = b - self.vt.T.dot(n)
        r_b = utils.frobenius_norm(q)
        q = np.atleast_2d(q) / r_b

        u_a = np.append(n, [0])
        v_b = np.append(n, [r_b])

        k = np.dot(np.diag(np.append(self.s, [0])),
                   np.identity(self.s.size + 1) - np.outer(u_a, v_b))

        inner_u, s_new, inner_vt = np.linalg.svd(k)

        if s_new.size > self.s.size:
            p = np.zeros((self.u.shape[0], 1))
            self.u = np.hstack((self.u, p))
            self.vt = np.vstack((self.vt, q))
        self.u = np.dot(self.u, inner_u)
        self.s = s_new
        self.vt = np.dot(inner_vt, self.vt)

    def trim(self):
        orig_size = min(self.shape)
        self.u = self.u[:, :orig_size]
        self.s = self.s[:orig_size]
        self.vt = self.vt[:orig_size, :]
