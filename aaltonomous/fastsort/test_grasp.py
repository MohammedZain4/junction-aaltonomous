"""python -m unittest aaltonomous.fastsort.test_grasp   (numpy + opencv)"""
import unittest

import numpy as np

from aaltonomous.fastsort.grasp import classify_top

S = 0.045  # object size: cube side = cylinder diameter = triangle altitude


def grid(inside, n=60, half=0.04, noise=0.0008, seed=0):
    g = np.linspace(-half, half, n)
    X, Y = np.meshgrid(g, g)
    P = np.c_[X.ravel(), Y.ravel()]
    P = P[inside(P)]
    return P + np.random.default_rng(seed).normal(0, noise, P.shape) + [0.3, 0.05]


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def square(t):
    R = rot(-t)
    return lambda P: np.all(np.abs(P @ R.T) <= S / 2, axis=1)


def circle(P):
    return np.linalg.norm(P, axis=1) <= S / 2


def triangle(t):
    side = 2 * S / np.sqrt(3)
    V = np.array([[0, 2 * S / 3], [-side / 2, -S / 3], [side / 2, -S / 3]]) @ rot(t).T

    def inside(P):
        s = []
        for i in range(3):
            a, b = V[i], V[(i + 1) % 3]
            s.append((b[0] - a[0]) * (P[:, 1] - a[1]) - (b[1] - a[1]) * (P[:, 0] - a[0]))
        s = np.array(s)
        return np.all(s >= 0, axis=0) | np.all(s <= 0, axis=0)
    return inside, V


def axis_close(axes, want, tol_deg=8):
    want = np.asarray(want) / np.linalg.norm(want)
    return any(np.degrees(np.arccos(min(1.0, abs(float(a @ want))))) <= tol_deg for a in axes)


class TestClassify(unittest.TestCase):
    def test_cube_axes_follow_yaw(self):
        for deg in (0, 20, 37, -30):
            t = np.radians(deg)
            shape, axes = classify_top(grid(square(t)))
            self.assertEqual(shape, "cube", deg)
            self.assertTrue(axis_close(axes, [np.cos(t), np.sin(t)]), (deg, axes))
            self.assertTrue(axis_close(axes, [-np.sin(t), np.cos(t)]), (deg, axes))

    def test_cylinder(self):
        shape, axes = classify_top(grid(circle))
        self.assertEqual(shape, "cylinder")
        self.assertEqual(axes, [])

    def test_triangle_axes_parallel_to_sides(self):
        for deg in (0, 25, -40):
            inside, V = triangle(np.radians(deg))
            shape, axes = classify_top(grid(inside))
            self.assertEqual(shape, "triangle", deg)
            self.assertEqual(len(axes), 3)
            for i in range(3):  # default pinch axes: parallel to each side
                self.assertTrue(axis_close(axes, V[(i + 1) % 3] - V[i]), (deg, i, axes))

    def test_lying_prism_ridge(self):
        # prism on its side: the top band (6 mm under the ridge, roof slope ~1.7) is a ~45 x 9 mm strip along the ridge
        for deg in (0, 30, -55, 90):
            t = np.radians(deg)
            R = rot(-t)

            def inside(P):
                Q = P @ R.T
                return (np.abs(Q[:, 0]) <= S / 2) & (np.abs(Q[:, 1]) <= 0.0045)
            shape, axes = classify_top(grid(inside, n=120))
            self.assertEqual(shape, "triangle", deg)
            self.assertEqual(len(axes), 1)
            self.assertTrue(axis_close(axes, [np.cos(t), np.sin(t)]), (deg, axes))   # pinch the two end faces

    def test_partial_cube_is_not_a_ridge(self):
        # a cube top half hidden (e.g. by an arm) is a 45 x 22 mm rectangle: aspect ~2 -> still a cube
        def inside(P):
            return (np.abs(P[:, 0]) <= S / 2) & (P[:, 1] >= 0) & (P[:, 1] <= S / 2)
        self.assertEqual(classify_top(grid(inside))[0], "cube")

    def test_too_few_points(self):
        self.assertEqual(classify_top(np.zeros((5, 2))), (None, []))


if __name__ == "__main__":
    unittest.main()
