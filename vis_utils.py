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
                 cmap='inferno', log=True,
                cbar_label=None, show_cbar=True, bounds=None, save=None, wind=None):

    df = df[df[values]>0]

    if bounds is None:
        bounds = (df[values].values.min(), df[values].values.max())
    
    if log:
        norm = LogNorm(vmin=bounds[0], vmax=bounds[1])
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

    if wind is not None:
        ax.arrow(0,0,wind[0], wind[1], 
                color="gray", width=300, 
                label="w: %1f m/s"%(np.sqrt(wind[0]**2 +\
                                    np.sqrt(wind[1]**2))))

    bg = ax.scatter(
        df["Easting"].values, df["Northing"].values, 
        c=df[values].values, s=150,
        vmin=bounds[0], vmax=bounds[1],
        norm=norm, edgecolor="k",
        cmap=cmap, alpha=.8)
    
    if show_cbar:
        cbar = fig.colorbar(bg, ax=ax)
        if cbar_label is not None:
            cbar.set_label(cbar_label)
        cbar.set_alpha(alpha=1)
        cbar.draw_all()
    if vent:
        ax.plot(vent[0], vent[1], 'k*', ms=15, label="Vent")
        ax.legend()

    ax.set_aspect("equal")

    return ax

def plot_residuals(df, vent=False, ax=None, values="Residual", title="Residual Plot", plot_type="size",
    cbar_label="% of Observation",  save=None, show_cbar=True, show_legend=True, wind=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
    else:
        fig = plt.gcf()
    

    ax.set_ylabel("Northing (m)")
    ax.set_xlabel("Easting (m)")
    ax.set_title(title)

    above = df[df[values]>1.02]
    justright = df[(df[values]>=.98) & (df[values]<=1.02)]
    below = df[df[values]<.98]

    if wind is not None:
        ax.arrow(0,0,wind[0], wind[1], 
                    color="gray", width=300, 
                    label="w: %1f m/s"%(np.sqrt(wind[0]**2 +\
                                        np.sqrt(wind[1]**2))))

    if plot_type == "size":
        ab = ax.scatter(above["Easting"].values, 
                   above["Northing"].values, 
                   s=np.log10(above[values].values)*500 + 20, 
                   c="r", marker="^", alpha=.5,
                  label="Over-estimated")
        jr = ax.scatter(justright["Easting"].values, 
                   justright["Northing"].values, 
                   s=64, 
                   c="g", marker="o", alpha=.5,
                  label="Well-estimated")
        bl = ax.scatter(below["Easting"].values, 
                   below["Northing"].values, 
                   s=np.log10((1/below[values]).values)*500 + 20, 
                   c="b", marker="v", alpha=.5,
                  label="Under-estimated")
        if show_legend:
            ax.legend(handles = [
                    plt.plot([], "k*", ms=10)[0],
                    plt.plot([], "r^", ms=np.sqrt(np.log10(2)*500 + 20), alpha=.5)[0],
                    plt.plot([], "r^", ms=np.sqrt(np.log10(1.5)*500 + 20), alpha=.5)[0],
                    plt.plot([], "r^", ms=np.sqrt(np.log10(1.01)*500 + 20), alpha=.5)[0],
                    plt.plot([], "go", ms=8, alpha=.5)[0],
                    plt.plot([], "bv", ms=np.sqrt(np.log10(1/.99)*500 + 20), alpha=.5)[0],
                    plt.plot([], "bv", ms=np.sqrt(np.log10(1/.75)*500 + 20), alpha=.5)[0],
                    plt.plot([], "bv", ms=np.sqrt(np.log10(1/.50)*500 + 20), alpha=.5)[0],
                ], labels=["Vent", 
                            "200%", 
                            "150%",
                            "102%",
                            "100%",
                            "98%",
                            "75%",
                            "50%"],
                loc='center left', bbox_to_anchor=(1, 0.5))

    elif plot_type == "cmap":
        ab = ax.scatter(above["Easting"].values, 
                   above["Northing"].values, 
                   s=150, 
                   c=above[values].values*100,
                   cmap="Reds",
                   edgecolor='red',
                   vmin=102, vmax = 200,
                   marker="^", alpha=1)
        jr = ax.scatter(justright["Easting"].values, 
                   justright["Northing"].values, 
                   s=120, 
                   c="g",
                   edgecolor='g',
                   marker="o", alpha=1)
        bl = ax.scatter(below["Easting"].values, 
                   below["Northing"].values, 
                   s=150,
                   c=below[values].values*100,
                   cmap="Blues_r",
                   edgecolor='blue',
                   vmin=50, vmax = 98,
                   marker="v", alpha=1)
        if show_cbar:
            ab_cbar = fig.colorbar(ab, ax=ax, label=cbar_label, extend="max")
            bl_cbar = fig.colorbar(bl, ax=ax, extend="min")
        if show_legend:
            ax.legend(handles = [
                plt.plot([], "k*", ms=15)[0],
                plt.plot([], "r^", ms=10, alpha=.6)[0],
                plt.plot([], "go", ms=10, alpha=.6)[0],
                plt.plot([], "bv", ms=10, alpha=.6)[0],
            ], labels=["Vent", "Over", "Good", "Under"])
    


    ax.plot(vent[0], vent[1], 'k*', ms=15)


    fig.tight_layout()

    ax.set_aspect("equal")

    return ax