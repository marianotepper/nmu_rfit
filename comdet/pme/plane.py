import numpy as np
import mpl_toolkits.mplot3d as plt3d
import matplotlib.colors as mpl_colors


class Plane(object):

    def __init__(self, data=None):
        self.eq = None
        if data is not None:
            self.fit(data)

    @property
    def min_sample_size(self):
        return 3

    def fit(self, data):
        if data.shape[0] < self.min_sample_size:
            raise ValueError('At least three points are needed to fit a plane')
        if data.shape[1] != 3:
            raise ValueError('Points must be 3D')

        data = np.hstack((data, np.ones((data.shape[0], 1))))
        if data.shape[0] == 3:
            data = np.vstack((data, np.zeros((1, 4))))
        _, _, v = np.linalg.svd(data, full_matrices=False)
        self.eq = v[3, :]

    def project(self, data):
        basis, x0 = self.point_and_basis()
        s = (data - x0).dot(basis.T)
        proj = x0 + s.dot(basis)
        return proj, s

    def point_and_basis(self):
        basis = self.basis()
        i_max = np.argmax(np.abs(self.eq[:2]))
        x0 = np.zeros((3,))
        x0[i_max] -= self.eq[3] / self.eq[i_max]
        return basis, x0

    def basis(self):
        n = self.eq[:3] / np.linalg.norm(self.eq[:3])
        basis1 = np.array([-n[1], n[0], 0])
        basis1 /= np.linalg.norm(basis1)
        basis2 = np.cross(n, basis1)
        basis2 /= np.linalg.norm(basis2)
        return np.vstack((basis1, basis2))

    def distances(self, data):
        if data.shape[1] == 3:
            data = np.hstack((data, np.ones((data.shape[0], 1))))
        return np.abs(np.dot(data, self.eq)) / np.linalg.norm(self.eq[:3])

    def _intersect(self, eq2):
        n1 = self.eq[:3] / np.linalg.norm(self.eq[:3])
        n2 = eq2[:3] / np.linalg.norm(eq2[:3])
        u = np.cross(n1, n2)

        i = np.argmax(np.abs(u))
        a = np.vstack((np.delete(self.eq, [i, 3], axis=0),
                       np.delete(eq2, [i, 3], axis=0)))
        b = np.array([-self.eq[3], -eq2[3]])

        x0 = np.linalg.solve(a, b)
        x0 = np.insert(x0, i, 0)
        return x0, u

    def _intersect_in_bounds(self, eq2, xlim, ylim, zlim):
        lower_bound = np.array([xlim[0], ylim[0], zlim[0]])
        upper_bound = np.array([xlim[1], ylim[1], zlim[1]])
        x0, u = self._intersect(eq2)

        mask = u > 0
        if np.any(mask):
            s_min1 = np.max((lower_bound[mask] - x0[mask]) / u[mask])
            s_max2 = np.min((upper_bound[mask] - x0[mask]) / u[mask])
        else:
            s_min1 = -np.inf
            s_max2 = np.inf
        mask = u < 0
        if np.any(mask):
            s_max1 = np.min((lower_bound[mask] - x0[mask]) / u[mask])
            s_min2 = np.max((upper_bound[mask] - x0[mask]) / u[mask])
        else:
            s_max1 = np.inf
            s_min2 = -np.inf
        s_min = np.fmax(s_min1, s_min2)
        s_max = np.fmin(s_max1, s_max2)
        p1 = x0 + s_min * u
        p2 = x0 + s_max * u
        if np.any(p1 < lower_bound) or np.any(p1 > upper_bound):
            p1 = None
        if np.any(p2 < lower_bound) or np.any(p2 > upper_bound):
            p2 = None
        return p1, p2

    def _sort_vertices(self, points):
        basis = self.basis()

        arr = np.array(points)
        arr = arr.dot(basis.T)
        center = np.mean(arr, axis=0)

        def compare(ta, tb):
            a = ta[1]
            b = tb[1]
            if b[0] < center[0] <= a[0]:
                return -1
            if a[0] < center[0] <= b[0]:
                return 1
            if a[0] == center[0] and b[0] == center[0]:
                if a[1] >= center[1] or b[1] >= center[1]:
                    if a[1] > b[1]:
                        return -1
                    else:
                        return 1
                if b[1] > a[1]:
                    return -1
                else:
                    return 1

            a2 = a - center
            b2 = b - center
            # compute the cross product of vectors
            # (center -> a) x (center -> b)
            det = np.cross(a2, b2)
            if det < 0:
                return -1
            if det > 0:
                return 1

            # points a and b are on the same line from the center
            # check which point is closer to the center
            if np.linalg.norm(a2) > np.linalg.norm(b2):
                return -1
            else:
                return 1

        seq = arr.tolist()
        idx = zip(*sorted(enumerate(seq), cmp=compare))[0]
        return [points[i] for i in idx]

    def plot_points(self, xlim, ylim, zlim):
        sides = [np.array([1., 0., 0., -xlim[0]]),
                 np.array([1., 0., 0., -xlim[1]]),
                 np.array([0., 1., 0., -ylim[0]]),
                 np.array([0., 1., 0., -ylim[1]]),
                 np.array([0., 0., 1., -zlim[0]]),
                 np.array([0., 0., 1., -zlim[1]])]

        points = []
        for pl in sides:
            p1, p2 = self._intersect_in_bounds(pl, xlim, ylim, zlim)
            if p1 is not None:
                points.append(p1)
            if p2 is not None:
                points.append(p2)

        if not points:
            return []
        else:
            return self._sort_vertices(points)

    def plot(self, ax, limits=None, color='b', alpha=0.5, **kwargs):
        if limits is None:
            xlim = list(ax.get_xlim())
            ylim = list(ax.get_ylim())
            zlim = list(ax.get_zlim())
        else:
            xlim = limits[0]
            ylim = limits[1]
            zlim = limits[2]

        points = self.plot_points(xlim, ylim, zlim)
        tri = plt3d.art3d.Poly3DCollection([points])
        tri.set_color(mpl_colors.colorConverter.to_rgba(color, alpha=alpha))
        tri.set_edgecolor(mpl_colors.colorConverter.to_rgba('k', alpha=0))
        ax.add_collection3d(tri)


def _test():
    import matplotlib.pyplot as plt

    x1 = np.array([[0, 1, 0], [1, .5, 0], [1, .2, 1]])
    l1 = Plane(x1)
    # np.random.seed(0)
    x2 = np.random.rand(3, 3)
    l2 = Plane(x2)
    # print x2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.axis('equal')

    # ax.scatter(x1[:, 0], x1[:, 1], x1[:, 2], c='b')
    ax.scatter(x2[:, 0], x2[:, 1], x2[:, 2], c='b')

    # limits = (list(ax.get_xlim()), list(ax.get_ylim()), list(ax.get_zlim()))

    # l1.plot(ax, color='g', alpha=0.2)
    l2.plot(ax, color='g', alpha=0.2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

if __name__ == '__main__':
    _test()
