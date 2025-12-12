#this program was run in VSCode
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter
from tkinter import *


def generate_terrain(n=300, smooth_sigma=3):
    x = np.linspace(-10, 10, n)
    y = np.linspace(-10, 10, n)
    X, Y = np.meshgrid(x, y)
    Z = np.cos(X / 2) + np.sin(Y / 4)
    Z = gaussian_filter(Z, smooth_sigma)
    return X, Y, Z

def find_local_maxima(Z):
    neigh = maximum_filter(Z, size=3, mode='nearest')
    mask = (Z == neigh)
    mask &= (Z >= neigh - 1e-12)
    coords = np.argwhere(mask)
    return [tuple(c) for c in coords]


def trace_river(Z, start, step_limit=2000):
    path = [start]
    nrows, ncols = Z.shape

    for _ in range(step_limit):
        r, c = path[-1]
        if r <= 0 or r >= nrows - 1 or c <= 0 or c >= ncols - 1:
            break

        local = Z[r-1:r+2, c-1:c+2]
        idx = np.unravel_index(np.argmin(local), (3, 3))
        nr, nc = r + idx[0] - 1, c + idx[1] - 1

        if Z[nr, nc] >= Z[r, c]:
            break

        path.append((nr, nc))
    return path

def plane_from_points(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    if np.linalg.norm(n) == 0:
        raise ValueError("Points are collinear; cannot define a plane.")
    return n / np.linalg.norm(n)

def strike_and_dip_from_normal(n):
    nx, ny, nz = n

    dip = np.degrees(np.arctan2(np.sqrt(nx**2 + ny**2), abs(nz)))

    strike = np.degrees(np.arctan2(ny, nx)) - 90.0

    strike %= 360.0

    return strike, dip

clicked_points = []
X = Y = Z = None  

def onclick(event):
    global clicked_points, X, Y, Z

    if event.inaxes is None:
        return

    x_click, y_click = event.xdata, event.ydata
    dist = (X - x_click)**2 + (Y - y_click)**2
    r, c = np.unravel_index(np.argmin(dist), dist.shape)

    p = np.array([X[r, c], Y[r, c], Z[r, c]])
    clicked_points.append(p)

    event.inaxes.plot(p[0], p[1], 'ro')
    event.inaxes.figure.canvas.draw()

    if len(clicked_points) == 3:
        p1, p2, p3 = clicked_points
        n = plane_from_points(p1, p2, p3)
        strike, dip = strike_and_dip_from_normal(n)

        strike_dip_label.config(
            text=f"Strike: {strike:.2f}°\nDip: {dip:.2f}°"
        )

        clicked_points = []

def plot_contour_with_rivers():
    maxima = find_local_maxima(Z)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cp = ax.contour(X, Y, Z, levels=30)
    ax.clabel(cp, inline=True, fontsize=8)

    for r, c in maxima:
        river = trace_river(Z, (r, c))
        ax.plot(X[r, c], Y[r, c], 'X')
        if len(river) > 1:
            ys = [Y[p] for p in river]
            xs = [X[p] for p in river]
            ax.plot(xs, ys, linewidth=2)

    ax.set_title("Contour Map with Simulated River Paths\nClick 3 points to compute strike & dip")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.tight_layout()
    plt.show()

def run_program():
    global X, Y, Z
    strike_dip_label.config(text="Strike: —\nDip: —")
    X, Y, Z = generate_terrain()
    plot_contour_with_rivers()

root = Tk()
root.title("Terrain Program")

run_button = Button(root, text="Run", command=run_program, width=20)
run_button.pack(pady=10)

strike_dip_label = Label(root, text="Strike: —\nDip: —", font=("Arial", 12))
strike_dip_label.pack(pady=10)

root.mainloop()