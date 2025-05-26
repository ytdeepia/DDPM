# kde_contours.py
from manim import *
import numpy as np
from scipy.stats import gaussian_kde
from skimage import measure


class KDEContours(VMobject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generate_contours()

    def generate_contours(self):
        np.random.seed(0)
        # Generate sample data: two clusters and a ring
        cluster1 = np.random.normal(loc=-2, scale=0.5, size=(200, 2))
        cluster2 = np.random.normal(loc=2, scale=0.7, size=(200, 2))
        angles = np.random.uniform(0, 2 * np.pi, 200)
        radii = 3 + np.random.normal(scale=0.2, size=200)
        ring = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        data = np.vstack([cluster1, cluster2, ring]).T  # shape becomes (2, N)

        # Compute the KDE over a grid
        kde = gaussian_kde(data)
        x_min, x_max = data[0].min() - 1, data[0].max() + 1
        y_min, y_max = data[1].min() - 1, data[1].max() + 1
        grid_size = 100
        xgrid = np.linspace(x_min, x_max, grid_size)
        ygrid = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(xgrid, ygrid)
        grid_coords = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(grid_coords).reshape(X.shape)

        # Define contour levels between the 50th and 99th percentiles
        level_min = np.percentile(Z, 50)
        level_max = np.percentile(Z, 99)
        levels = np.linspace(level_min, level_max, 6)

        # Create a group for the contours
        contours_group = VGroup()
        for lvl in levels:
            contours = measure.find_contours(Z, lvl)
            for contour in contours:
                points = []
                for point in contour:
                    # Convert grid indices to data coordinates
                    x_coord = x_min + (point[1] / (grid_size - 1)) * (x_max - x_min)
                    y_coord = y_min + (point[0] / (grid_size - 1)) * (y_max - y_min)
                    points.append(np.array([x_coord, y_coord, 0]))
                contour_mobject = VMobject()
                contour_mobject.set_points_as_corners(points)
                # Map the contour level to a color gradient from BLUE_E to RED
                color_value = (lvl - level_min) / (level_max - level_min)
                contour_color = interpolate_color(BLUE_E, RED, color_value)
                contour_mobject.set_stroke(color=contour_color, width=2)
                contours_group.add(contour_mobject)
        self.add(contours_group)
