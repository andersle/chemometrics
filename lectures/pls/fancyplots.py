"""Helper methods for plotting."""

import numpy as np
import seaborn as sns
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool, Label, Span
from bokeh.palettes import Colorblind8
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

sns.set_theme(style="ticks", context="notebook", palette="muted")


def _add_yhat_vs_y(ax, model, X, y, title=""):
    """Add plot of y vs ŷ to given axis."""
    y_hat = model.predict(X)
    score = r2_score(y, y_hat)
    ax.scatter(
        y,
        y_hat,
        label=f"R² = {score:.5g}",
    )
    ax.set(xlabel="y", ylabel="ŷ", title=title)
    ax.legend()


def mpl_yhat_vs_y(model, X_train, y_train, X_test=None, y_test=None, title=""):
    """Plot predicted vs actual values for the given model."""
    if X_test is None or y_test is None:
        fig, ax1 = plt.subplots(constrained_layout=True, figsize=(4, 4))
        ax2 = None
    else:
        fig, (ax1, ax2) = plt.subplots(
            constrained_layout=True, ncols=2, figsize=(8, 4)
        )
    _add_yhat_vs_y(ax1, model, X_train, y_train, title=f"{title} (train)")
    if X_test is not None and y_test is not None and ax2 is not None:
        _add_yhat_vs_y(ax2, model, X_test, y_test, title=f"{title} (test)")
    sns.despine(fig=fig)
    return [fig]


def mpl_show_coefficients(model, xvars, yvars, sort=False):
    """Plot coefficients from a regression model.

    Parameters
    ----------
    model : object like sklearn.base.BaseEstimator
        A sklearn model. Assumed to have coefficients.
    xvars : list of strings
        The name of the x variables.
    yvars : list of strings
        The name of the y variables.
    sort : boolean
        If True, the coefficients are sorted by their absolute value.

    Returns
    -------
    figures : list of object like matplotlib.backend_bases.FigureCanvasBase
        The figures created here (one for each y-variable).

    """
    figures = []
    for yvari in yvars:
        fig, ax = plt.subplots(constrained_layout=True)
        pos = np.arange(1, len(xvars) + 1)
        coef = model.coef_.flatten()
        if sort:
            idx = np.argsort(abs(coef))
            coef = coef[idx]
        ax.plot(pos, coef, marker="o")
        ax.axhline(y=0, ls=":", color="black")
        ax.set(
            xlabel="X-variables",
            ylabel="Coefficient",
            title=f"Coefficients for {yvari}",
        )
        sns.despine(fig=fig)
        figures.append(fig)
    return figures


def mpl_plot_coefficients(model, xvars, yvars, variable_type=None):
    """Plot coefficients from a PLS model.

    Parameters
    ----------
    model : object like sklearn.base.BaseEstimator
        A sklearn model. Assumed to have coefficients.
    xvars : list of strings
        The name of the x variables.
    yvars : list of strings
        The name of the y variables.
    variable_type : dict of strings, optional
        Types of variables. If given, it will be used to
        indicate the type of variable via colors.

    Returns
    -------
    figures : list of objects like matplotlib.backend_bases.FigureCanvasBase
        A list of figures created here (one for each y-variable).

    """
    figures = []
    for i, yvari in enumerate(yvars):
        fig, ax = plt.subplots(constrained_layout=True)
        pos = np.arange(1, len(xvars) + 1)
        coef = model.coef_[i, :]
        if variable_type is None:
            ax.bar(pos, coef)
        else:
            types = [variable_type[i] for i in xvars]
            for typei in set(types):
                idx = [j for j, typej in enumerate(types) if typej == typei]
                ax.bar(pos[idx], coef[idx], label=typei)
            ax.legend()
        ax.axhline(y=0, ls=":", color="black")
        ax.set(
            xlabel="X-variables",
            ylabel="Coefficient",
            title=f"Coefficients for {yvari}",
        )
        sns.despine(fig=fig)
        figures.append(fig)
    return figures


def bokeh_plot_coefficients(
    model, xvars, yvars, variable_type=None, description=None
):
    """Plot coefficients from a PLS model using bokeh.

    Parameters
    ----------
    model : object like sklearn.base.BaseEstimator
        A sklearn model. Assumed to have coefficients.
    xvars : list of strings
        The name of the x variables.
    yvars : list of strings
        The name of the y variables.
    variable_type : dict of strings, optional
        Types of variables. If given, it will be used to
        indicate the type of variable via colors.
    description : dict of strings, optional
        Description of variables. If given, it will be used
        to show a description of the variables.

    Returns
    -------
    figures : object like bokeh.models.layouts.Column
        The figures created here (one for each y-variable).

    """
    figures = []
    if variable_type is None:
        types = ["Unknown" for i in xvars]
    else:
        types = [variable_type[i] for i in xvars]
    if description is None:
        desc = ["Variable" for i in xvars]
    else:
        desc = [description[i] for i in xvars]

    for i, yvari in enumerate(yvars):
        # Arrange data for bokeh:
        pos = np.arange(1, len(xvars) + 1)
        coef = model.coef_[i, :]

        source = ColumnDataSource(
            data={
                "xvars": xvars,
                "types": types,
                "pos": pos,
                "coef": coef,
                "desc": desc,
            }
        )

        p = figure(title=f"Coefficients for {yvari}")
        hline = Span(
            location=0,
            dimension="width",
            line_color="black",
            line_width=2,
            line_dash="dotted",
        )
        p.renderers.extend([hline])
        p.vbar(
            x="pos",
            top="coef",
            width=0.9,
            source=source,
            legend_field="types",
            line_color="white",
            fill_color=factor_cmap(
                "types", palette=Colorblind8, factors=list(set(types))
            ),
        )
        p.add_tools(
            HoverTool(
                tooltips=[
                    ("Variable", "@xvars"),
                    ("Coef", "@coef"),
                    ("Type", "@types"),
                    ("Description", "@desc"),
                ],
            )
        )
        p.xaxis.axis_label = "X-variables"
        p.yaxis.axis_label = "Coefficients"
        figures.append(p)
    return column(*figures)


def _select_loadings(model, x_type, y_type):
    """Select loadings to use for the given PLS model."""
    if x_type == "loadings":
        loadingsx = model.x_loadings_
    elif x_type == "weights":
        loadingsx = model.x_weights_
    else:
        loadingsx = model.x_rotations_

    if y_type == "loadings":
        loadingsy = model.y_loadings_
    elif y_type == "weights":
        loadingsy = model.y_weights_
    else:
        loadingsy = model.y_rotations_
    return loadingsx, loadingsy


def mpl_plot_loadings(
    model,
    xvars,
    yvars,
    variable_type=None,
    idx1=0,
    idx2=1,
    factor=1.0,
    x_type="rotations",
    y_type="loadings",
    xylim=None,
):
    """Plot loadings for a PLS model.

    Parameters
    ----------
    model : object like sklearn.base.BaseEstimator
        A sklearn model. Assumed to have coefficients.
    xvars : list of strings
        The name of the x variables.
    yvars : list of strings
        The name of the y variables.
    variable_type : dict of strings
        Types of variables.
    idx1 : integer, optional
        For selecting the latent variable to use for the x-axis.
    idx2 : integer, optional
        For selecting the latent variable to use for the y-axis.
    factor : float, optional
        For adjusting the positing of the y-loadings

    Returns
    -------
    figure : object like matplotlib.backend_bases.FigureCanvasBase
        The figure created here.
    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 6))
    loadingsx, loadingsy = _select_loadings(model, x_type, y_type)

    if variable_type is None:
        types = [i for i in xvars]
    else:
        types = [variable_type[i] for i in xvars]
    for typei in set(types):
        idx = [j for j, typej in enumerate(types) if typej == typei]
        ax.scatter(
            loadingsx[idx, idx1],
            loadingsx[idx, idx2],
            label=typei,
            s=75,
            edgecolor="black",
        )

    for i, yvar in enumerate(yvars):
        ax.plot(
            [0, factor * loadingsy[i, idx1]],
            [0, factor * loadingsy[i, idx2]],
            color="red",
        )
        ax.text(
            factor * loadingsy[i, idx1],
            factor * loadingsy[i, idx2],
            yvar,
            color="red",
            va="bottom" if loadingsy[i, idx2] > 0 else "top",
            ha="center",
        )
    ax.axhline(y=0, color="k", ls=":")
    ax.axvline(x=0, color="k", ls=":")
    ax.set_aspect("equal")
    if xylim is None:
        ax.set_xlim(-0.4, 0.4)
        ax.set_ylim(-0.4, 0.4)
    else:
        ax.set_xlim(xylim)
        ax.set_ylim(xylim)
    ax.set(xlabel=f"PLS component {idx1+1}", ylabel=f"PLS component {idx2+1}")
    ax.set_title("Loadings", loc="left")
    ax.legend()
    sns.despine(fig=fig)
    return [fig]


def bokeh_plot_loadings(
    model,
    xvars,
    yvars,
    variable_type,
    description,
    idx1=0,
    idx2=1,
    factor=1.0,
    x_type="rotations",
    y_type="loadings",
):
    """Plot loadings for a PLS model with bokeh.

    Parameters
    ----------
    model : object like sklearn.base.BaseEstimator
        A sklearn model. Assumed to have coefficients.
    xvars : list of strings
        The name of the x variables.
    yvars : list of strings
        The name of the y variables.
    variable_type : dict of strings
        Types of variables.
    description : dict of strings
        Description of variables.
    idx1 : integer, optional
        For selecting the latent variable to use for the x-axis.
    idx2 : integer, optional
        For selecting the latent variable to use for the y-axis.
    factor : float, optional
        For adjusting the positing of the y-loadings

    Returns
    -------
    figure : object like matplotlib.backend_bases.FigureCanvasBase
        The figure created here.
    """
    p = figure()

    loadingsx, loadingsy = _select_loadings(model, x_type, y_type)

    types = [variable_type[i] for i in xvars]

    source_x = ColumnDataSource(
        data={
            "x": loadingsx[:, 0],
            "y": loadingsx[:, 1],
            "types": types,
            "desc": [description[i] for i in xvars],
            "xvars": xvars,
        }
    )

    xfactor = loadingsy[:, 0] * factor
    yfactor = loadingsy[:, 1] * factor

    source_y = ColumnDataSource(
        data={
            "x": xfactor,
            "y": yfactor,
            "types": [variable_type[i] for i in yvars],
            "desc": [description[i] for i in yvars],
            "xvars": yvars,
        }
    )

    p = figure(title="Loadings", x_range=(-0.4, 0.4), y_range=(-0.4, 0.4))

    for i, yvari in enumerate(yvars):
        p.line(
            [0, xfactor[i]],
            [0, yfactor[i]],
            line_width=4,
            color="red",
        )
        txt = Label(
            x=xfactor[i],
            y=yfactor[i],
            text=yvari,
            text_color="red",
            text_align="center",
            text_baseline="bottom" if yfactor[i] > 0 else "top",
        )
        p.add_layout(txt)

    vline = Span(
        location=0,
        dimension="height",
        line_color="black",
        line_width=2,
        line_dash="dotted",
    )
    hline = Span(
        location=0,
        dimension="width",
        line_color="black",
        line_width=2,
        line_dash="dotted",
    )
    p.renderers.extend([vline, hline])

    p.scatter(
        x="x",
        y="y",
        size=16,
        legend_group="types",
        line_color="black",
        fill_color=factor_cmap(
            "types", palette=Colorblind8, factors=list(set(types))
        ),
        name="x-loadings",
        source=source_x,
    )

    p.scatter(
        x="x",
        y="y",
        size=8,
        color="red",
        line_color="red",
        marker="circle",
        name="y-loadings",
        source=source_y,
    )

    p.add_tools(
        HoverTool(
            tooltips=[
                ("Variable", "@xvars"),
                ("Type", "@types"),
                ("Description", "@desc"),
            ],
        )
    )

    p.xaxis.axis_label = f"PLS component {idx1+1}"
    p.yaxis.axis_label = f"PLS component {idx2+1}"
    return p
