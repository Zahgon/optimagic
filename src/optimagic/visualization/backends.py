import itertools
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload, runtime_checkable

import numpy as np
import plotly.graph_objects as go

from optimagic.config import (
    IS_ALTAIR_INSTALLED,
    IS_BOKEH_INSTALLED,
    IS_MATPLOTLIB_INSTALLED,
)
from optimagic.exceptions import InvalidPlottingBackendError, NotInstalledError
from optimagic.visualization.plotting_utilities import LineData, MarkerData

if TYPE_CHECKING:
    import altair as alt
    import bokeh
    import matplotlib.pyplot as plt


@runtime_checkable
class LinePlotFunction(Protocol):
    def __call__(
        self,
        lines: list[LineData],
        *,
        title: str | None,
        xlabel: str | None,
        xrange: tuple[float, float] | None,
        ylabel: str | None,
        yrange: tuple[float, float] | None,
        template: str | None,
        height: int | None,
        width: int | None,
        legend_properties: dict[str, Any] | None,
        margin_properties: dict[str, Any] | None,
        horizontal_line: float | None,
        marker: MarkerData | None,
        subplot: Any | None = None,
    ) -> Any:
        """Protocol of the line_plot function used for type checking.

        Args:
            ...: All other argument descriptions can be found in the docstring of the
                `line_plot` function.
            subplot: The subplot to which the lines should be plotted. The type of this
                argument depends on the backend used. If not provided, a new figure is
                created.

        """
        ...


@runtime_checkable
class GridLinePlotFunction(Protocol):
    def __call__(
        self,
        lines_list: list[list[LineData]],
        *,
        n_rows: int,
        n_cols: int,
        titles: list[str] | None,
        xlabels: list[str] | None,
        xrange: tuple[float, float] | None,
        share_x: bool,
        ylabels: list[str] | None,
        yrange: tuple[float, float] | None,
        share_y: bool,
        template: str | None,
        height: int | None,
        width: int | None,
        legend_properties: dict[str, Any] | None,
        margin_properties: dict[str, Any] | None,
        plot_title: str | None,
        marker_list: list[MarkerData] | None,
        make_subplot_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Protocol of the grid_line_plot function used for type checking.

        Args:
            ...: All other argument descriptions can be found in the docstring of the
                `grid_line_plot` function.

        """
        ...


def _line_plot_plotly(
    lines: list[LineData],
    *,
    title: str | None,
    xlabel: str | None,
    xrange: tuple[float, float] | None,
    ylabel: str | None,
    yrange: tuple[float, float] | None,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    horizontal_line: float | None,
    marker: MarkerData | None,
    subplot: tuple[go.Figure, int, int] | None = None,
) -> go.Figure:
    pass


def _grid_line_plot_plotly(
    lines_list: list[list[LineData]],
    *,
    n_rows: int,
    n_cols: int,
    titles: list[str] | None,
    xlabels: list[str] | None,
    xrange: tuple[float, float] | None,
    share_x: bool,
    ylabels: list[str] | None,
    yrange: tuple[float, float] | None,
    share_y: bool,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    plot_title: str | None,
    marker_list: list[MarkerData] | None,
    make_subplot_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    pass


def _line_plot_matplotlib(
    lines: list[LineData],
    *,
    title: str | None,
    xlabel: str | None,
    xrange: tuple[float, float] | None,
    ylabel: str | None,
    yrange: tuple[float, float] | None,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    horizontal_line: float | None,
    marker: MarkerData | None,
    subplot: "plt.Axes | None" = None,
) -> "plt.Axes":
    pass


def _grid_line_plot_matplotlib(
    lines_list: list[list[LineData]],
    *,
    n_rows: int,
    n_cols: int,
    titles: list[str] | None,
    xlabels: list[str] | None,
    xrange: tuple[float, float] | None,
    share_x: bool,
    ylabels: list[str] | None,
    yrange: tuple[float, float] | None,
    share_y: bool,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    plot_title: str | None,
    marker_list: list[MarkerData] | None,
    make_subplot_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    pass


def _line_plot_bokeh(
    lines: list[LineData],
    *,
    title: str | None,
    xlabel: str | None,
    xrange: tuple[float, float] | None,
    ylabel: str | None,
    yrange: tuple[float, float] | None,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    horizontal_line: float | None,
    marker: MarkerData | None,
    subplot: "bokeh.plotting.figure | None" = None,
) -> "bokeh.plotting.figure":
    pass


def _grid_line_plot_bokeh(
    lines_list: list[list[LineData]],
    *,
    n_rows: int,
    n_cols: int,
    titles: list[str] | None,
    xlabels: list[str] | None,
    xrange: tuple[float, float] | None,
    share_x: bool,
    ylabels: list[str] | None,
    yrange: tuple[float, float] | None,
    share_y: bool,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    plot_title: str | None,
    marker_list: list[MarkerData] | None,
    make_subplot_kwargs: dict[str, Any] | None = None,
) -> "bokeh.models.GridPlot":
    pass


def _line_plot_altair(
    lines: list[LineData],
    *,
    title: str | None,
    xlabel: str | None,
    xrange: tuple[float, float] | None,
    ylabel: str | None,
    yrange: tuple[float, float] | None,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    horizontal_line: float | None,
    marker: MarkerData | None,
    subplot: None = None,
) -> "alt.Chart":
    pass


def _grid_line_plot_altair(
    lines_list: list[list[LineData]],
    *,
    n_rows: int,
    n_cols: int,
    titles: list[str] | None,
    xlabels: list[str] | None,
    xrange: tuple[float, float] | None,
    share_x: bool,
    ylabels: list[str] | None,
    yrange: tuple[float, float] | None,
    share_y: bool,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    plot_title: str | None,
    marker_list: list[MarkerData] | None,
    make_subplot_kwargs: dict[str, Any] | None = None,
) -> "alt.Chart | alt.HConcatChart | alt.VConcatChart":
    pass


def line_plot(
    lines: list[LineData],
    backend: Literal["plotly", "matplotlib", "bokeh", "altair"] = "plotly",
    *,
    title: str | None = None,
    xlabel: str | None = None,
    xrange: tuple[float, float] | None = None,
    ylabel: str | None = None,
    yrange: tuple[float, float] | None = None,
    template: str | None = None,
    height: int | None = None,
    width: int | None = None,
    legend_properties: dict[str, Any] | None = None,
    margin_properties: dict[str, Any] | None = None,
    horizontal_line: float | None = None,
    marker: MarkerData | None = None,
) -> Any:
    """Create a line plot corresponding to the specified backend.

    Args:
        lines: List of objects each containing data for a line in the plot.
            The order of lines in the list determines the order in which they are
            plotted, with later lines being rendered on top of earlier ones.
        backend: The backend to use for plotting.
        title: Title of the plot.
        xlabel: Label for the x-axis.
        xrange: View limits for the x-axis.
        ylabel: Label for the y-axis.
        yrange: View limits for the y-axis.
        template: Backend-specific template for styling the plot.
        height: Height of the plot (in pixels).
        width: Width of the plot (in pixels).
        legend_properties: Backend-specific properties for the legend.
        margin_properties: Backend-specific properties for the plot margins.
        horizontal_line: If provided, a horizontal line is drawn at the specified
            y-value.
        marker: An object containing data for a marker in the plot.

    Returns:
        A figure object corresponding to the specified backend.

    """
    _line_plot_backend_function = _get_plot_function(backend, grid_plot=False)

    fig = _line_plot_backend_function(
        lines,
        title=title,
        xlabel=xlabel,
        xrange=xrange,
        ylabel=ylabel,
        yrange=yrange,
        template=template,
        height=height,
        width=width,
        legend_properties=legend_properties,
        margin_properties=margin_properties,
        horizontal_line=horizontal_line,
        marker=marker,
    )

    return fig


def grid_line_plot(
    lines_list: list[list[LineData]],
    backend: Literal["plotly", "matplotlib", "bokeh", "altair"] = "plotly",
    *,
    n_rows: int,
    n_cols: int,
    titles: list[str] | None = None,
    xlabels: list[str] | None = None,
    xrange: tuple[float, float] | None = None,
    share_x: bool = False,
    ylabels: list[str] | None = None,
    yrange: tuple[float, float] | None = None,
    share_y: bool = False,
    template: str | None = None,
    height: int | None = None,
    width: int | None = None,
    legend_properties: dict[str, Any] | None = None,
    margin_properties: dict[str, Any] | None = None,
    plot_title: str | None = None,
    marker_list: list[MarkerData] | None = None,
    make_subplot_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Create a grid of line plots corresponding to the specified backend.

    Args:
        lines_list: A list where each element is a list of objects containing data
            for the lines in a subplot. The order of sublists determines the order
            of subplots in the grid (row-wise), and the order of lines within each
            sublist determines the order of lines in that subplot.
        backend: The backend to use for plotting.
        n_rows: Number of rows in the grid.
        n_cols: Number of columns in the grid.
        titles: Titles for each subplot in the grid.
        xlabels: Labels for the x-axis of each subplot.
        xrange: View limits for the x-axis of each subplot.
        share_x: If True, all subplots share the same x-axis limits and each subplot in
            a column actually share the x-axis.
        ylabels: Labels for the y-axis of each subplot.
        yrange: View limits for the y-axis of each subplot.
        share_y: If True, all subplots share the same y-axis limits and each subplot in
            a row actually share the y-axis.
        template: Backend-specific template for styling the plots.
        height: Height of the entire grid plot (in pixels).
        width: Width of the entire grid plot (in pixels).
        legend_properties: Backend-specific properties for the legend.
        margin_properties: Backend-specific properties for the plot margins.
        plot_title: Title for the entire grid plot.
        marker_list: A list where where each element is an object containing data
            for a marker in a subplot. The order of objects in the list determines
            the subplot on which the marker is plotted.

    Returns:
        A figure object corresponding to the specified backend.

    """
    _grid_line_plot_backend_function = _get_plot_function(backend, grid_plot=True)

    fig = _grid_line_plot_backend_function(
        lines_list,
        n_rows=n_rows,
        n_cols=n_cols,
        titles=titles,
        xlabels=xlabels,
        xrange=xrange,
        share_x=share_x,
        ylabels=ylabels,
        yrange=yrange,
        share_y=share_y,
        template=template,
        height=height,
        width=width,
        legend_properties=legend_properties,
        margin_properties=margin_properties,
        plot_title=plot_title,
        marker_list=marker_list,
        make_subplot_kwargs=make_subplot_kwargs,
    )

    return fig


BACKEND_AVAILABILITY_AND_LINE_PLOT_FUNCTION: dict[
    str, tuple[bool, LinePlotFunction, GridLinePlotFunction]
] = {
    "plotly": (True, _line_plot_plotly, _grid_line_plot_plotly),
    "matplotlib": (
        IS_MATPLOTLIB_INSTALLED,
        _line_plot_matplotlib,
        _grid_line_plot_matplotlib,
    ),
    "bokeh": (
        IS_BOKEH_INSTALLED,
        _line_plot_bokeh,
        _grid_line_plot_bokeh,
    ),
    "altair": (
        IS_ALTAIR_INSTALLED,
        _line_plot_altair,
        _grid_line_plot_altair,
    ),
}


@overload
def _get_plot_function(
    backend: Literal["plotly", "matplotlib", "bokeh", "altair"],
    grid_plot: Literal[False],
) -> LinePlotFunction: ...


@overload
def _get_plot_function(
    backend: Literal["plotly", "matplotlib", "bokeh", "altair"],
    grid_plot: Literal[True],
) -> GridLinePlotFunction: ...


def _get_plot_function(
    backend: str, grid_plot: bool
) -> LinePlotFunction | GridLinePlotFunction:
    if backend not in BACKEND_AVAILABILITY_AND_LINE_PLOT_FUNCTION:
        msg = (
            f"Invalid plotting backend '{backend}'. "
            f"Available backends: "
            f"{', '.join(BACKEND_AVAILABILITY_AND_LINE_PLOT_FUNCTION.keys())}"
        )
        raise InvalidPlottingBackendError(msg)

    (
        _is_backend_available,
        _line_plot_backend_function,
        _grid_line_plot_backend_function,
    ) = BACKEND_AVAILABILITY_AND_LINE_PLOT_FUNCTION[backend]

    if not _is_backend_available:
        msg = (
            f"The {backend} backend is not installed. "
            f"Install the package using either 'pip install {backend}' or "
            f"'conda install -c conda-forge {backend}'"
        )
        raise NotInstalledError(msg)

    if grid_plot:
        return _grid_line_plot_backend_function
    else:
        return _line_plot_backend_function
