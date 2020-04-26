import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as pe

def plot_contour(df, vent=False, ax=None, values="MassArea", title="Isomass Plot", cmap='inferno', log=True, lines=True, line_cmap=None, line_colors=None, background="gradient", cbar_label=None,  save=None):

    # TODO: Figure out why low phi deposits break both the negative/grayscale linemaps, and the filled contours.  

    df = df[df[values]>0]

    piv = pd.pivot_table(df, index="Northing",
                         columns="Easting", values=values)

    if log:
        norm = LogNorm(vmin=piv.values.min(), vmax=piv.values.max())
    else:
        norm = None


    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = plt.gcf()
    # ax.axis('equal')
    ax.set_ylabel("Northing")
    ax.set_xlabel("Easting")
    ax.set_title(title)

    if background == "gradient":
        if log:
            vals = np.log10(piv.values)
        else:
            vals = piv.values
        bg = ax.imshow(vals,
                       extent=[piv.columns.min(), piv.columns.max(),
                               piv.index.min(), piv.index.max()],
                       origin='lower',
                       cmap=cmap, alpha=1)
        if log:
            cbar = fig.colorbar(bg, ax=ax, format=r"$10^{%d}$")
        else:
            cbar = fig.colorbar(bg, ax=ax)
        cbar.set_alpha(alpha=1)
        cbar.draw_all()
        if cbar_label is not None:
            cbar.ax.set_ylabel(cbar_label, rotation=270)
    elif background == "fill":
        bg = ax.contourf(piv.columns, piv.index, piv.values,
                         cmap=cmap, vmin=piv.values.min(), vmax=piv.values.max())
        cbar = fig.colorbar(bg, ax=ax)
        cbar.set_alpha(alpha=1)
        cbar.draw_all()
        if cbar_label is not None:
            cbar.ax.set_ylabel(cbar_label, rotation=270)

    if lines:
        if line_cmap == "grayscale":
            colormap = cm.get_cmap(cmap, 256)
            oldcolors = colormap(np.linspace(0, 1, 256))
            newcolors = []
            for i, col in enumerate(oldcolors):
                r, g, b = [(1-col[j]) for j in range(3)]
                luminosity = 0.21*r + 0.72*g + 0.07*b
                newcolors.append([luminosity, luminosity, luminosity, 1])
            newcmp = ListedColormap(np.array(newcolors))
            lns = ax.contour(piv.columns, piv.index, piv.values, norm=norm,
                             cmap=newcmp)
        elif line_cmap == "negative":
            colormap = cm.get_cmap(cmap, 256)
            oldcolors = colormap(np.linspace(0, 1, 256))
            newcolors = [[(1-col[j]) for j in range(3)] + [1] for col in oldcolors]
            newcmp = ListedColormap(np.array(newcolors))
            lns = ax.contour(piv.columns, piv.index, piv.values, norm=norm,
                             cmap=newcmp)
        elif line_cmap is not None:
            lns = ax.contour(piv.columns, piv.index, piv.values, norm=norm,
                             cmap=line_cmap)
        elif line_colors is not None:
            lns = ax.contour(piv.columns, piv.index, piv.values, norm=norm,
                             colors=line_colors)
        else:
            lns = ax.contour(piv.columns, piv.index, piv.values, norm=norm,
                             colors='k')
        fmt = ticker.LogFormatterMathtext()
        fmt.create_dummy_axis()
        ax.clabel(lns, lns.levels, fmt=fmt, fontsize=10)



    if vent:
        ax.plot(vent[0], vent[1], 'r^', ms=8)

    ax.set_xlim(right=piv.columns.max(), left=piv.columns.min())
    ax.set_ylim(bottom=piv.index.min(), top=piv.index.max())
    ax.set_aspect("equal")

    return ax

def plot_sample(df, vent=False, ax=None, values="MassArea", title="Isomass Plot", \
                 cmap='inferno', log=True, lines=True, line_cmap=None, line_colors=None, \
                 background="gradient", cbar_label=None,  save=None):

    # TODO: Figure out why low phi deposits break both the negative/grayscale linemaps, 
    # and the filled contours.  

    df = df[df[values]>0]

    piv = pd.pivot_table(df, index="Northing",
                         columns="Easting", values=values)

    if log:
        norm = LogNorm(vmin=piv.values.min(), vmax=piv.values.max())
    else:
        norm = None


    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = plt.gcf()
    # ax.axis('equal')
    ax.set_ylabel("Northing")
    ax.set_xlabel("Easting")
    ax.set_title(title)

    if log:
        vals = np.log10(df[values])
    else:
        vals = df[values]
    bg = ax.scatter(df["Easting"].values, df["Northing"].values, df[values].values, c=vals, cmap=cmap, alpha=.8)
#     if log:
#         cbar = fig.colorbar(bg, ax=ax, format=r"$10^{%d}$")
#     else:
#         cbar = fig.colorbar(bg, ax=ax)

    cbar = fig.colorbar(bg, ax=ax)
    if cbar_label is not None:
        cbar.set_label(cbar_label)
    cbar.set_alpha(alpha=1)
    cbar.draw_all()
    if vent:
        ax.plot(vent[0], vent[1], 'r^', ms=8)

    ax.set_aspect("equal")

    return ax