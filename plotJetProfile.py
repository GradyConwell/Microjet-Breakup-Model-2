# Grady Conwell, 2026. 
# Authored by Opus 5.5, Sonnet 5.5, with direction and material provided by Grady Conwell
#
# Jet profile figures: the upper half of the jet and its drops filled, the lower half as a dashed outline, on a
# true-scale r-z frame, optionally stacked as a time series, with a table of the run's parameters underneath.
# Text is black on the white background and turns white wherever it crosses a filled shape.
#
# Usage:
#   .venv/bin/python plotJetProfile.py                    window: pick a run and adjust time, span, domain, then Save
#   .venv/bin/python plotJetProfile.py results/<run>      the same window, opened on that run
#   .venv/bin/python plotJetProfile.py results/<run> --save [--start 300 --stop 320 --interval 4 | --single 320]
#                                                         [--domain 0 1000] [--format pdf]
# Every other knob is in Settings below.

import argparse
import glob
import math
import os
import re
import warnings
from dataclasses import dataclass, field, replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D, Bbox, ScaledTranslation
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider


@dataclass
class Settings:
    # ---- What to render -------------------------------------------------------------------------------------------
    domainStart: float = 0.0             # µm, left edge of the rendered stream (the nozzle is at 0)
    domainLength: float | None = 1000.0  # µm of stream to render; None = through the farthest drop
    radiusLimit: float = 50.0            # µm, the r axis runs from -radiusLimit to +radiusLimit

    # ---- Panels: a time series of jets stacked top to bottom ------------------------------------------------------
    panelize: bool = True                # False renders one frame, at startTime
    startTime: float | None = None       # µs; None = stopTime - 20 µs
    stopTime: float | None = None        # µs, last panel (inclusive); None = the run's last frame
    interval: float = 4.0                # µs between panels (frames are 1 µs apart; the nearest frame is used)
    timeOrigin: float | str = "start"    # time labels show t - origin: "start" (first panel), "absolute", or µs
    panelGap: float = 0.14               # in, vertical gap between panels

    # ---- Colors ---------------------------------------------------------------------------------------------------
    background: str = "white"
    fillColor: str = "#28587a"           # blue-sapphire (as in plotFrames.py): jet and drops, upper half
    satelliteColor: str | None = None    # drops smaller than satelliteRadius; None = fillColor
    satelliteRadius: float = 18.5        # µm
    outlineColor: str | None = None      # solid upper edge and dashed lower outline; None = fillColor
    textColor: str = "black"             # text on the background
    textOnFillColor: str = "white"       # the part of any text that lies on a filled shape
    axisColor: str = "black"

    # ---- Lines ----------------------------------------------------------------------------------------------------
    outlineWidth: float = 0.3            # pt
    dashedWidth: float = 0.3             # pt
    dashPattern: tuple = (0, (4, 3))     # (offset, (dash, gap)) in pt, independent of line weight
    axisWidth: float = 0.4               # pt
    majorTickLength: float = 4.0         # pt, ruler major ticks (downward); also the r-axis ticks
    minorTickLength: float = 2.0         # pt, ruler minor ticks (downward)
    arrowSize: float = 5.0               # pt, z-axis arrowhead

    # ---- Ruler: a z scale under the bottom panel, ticks hanging down from a bar -------------------------------------
    showRuler: bool = True
    rulerGap: float = 0.08               # in, between the bottom panel and the ruler bar
    majorTick: float = 100.0             # µm between labelled ticks
    minorTick: float = 25.0              # µm between unlabelled ticks (0 = none)
    lengthAxisLabel: str = "$z$ (µm)"    # under the ruler
    radiusAxisLabel: str = "$r$ (µm)"    # beside the +r axis
    radiusLabelPanels: str = "all"       # r-axis labels on "all" panels or only the "first" one
    timeLabelFormat: str = "t = {:+.1f} µs"

    # ---- Fonts ----------------------------------------------------------------------------------------------------
    fontFamilies: list = field(default_factory=lambda: ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"])
    fontSize: float = 7.0                # pt, tick and axis labels
    timeLabelSize: float = 8.0           # pt
    tableFontSize: float = 7.0           # pt

    # ---- Geometry -------------------------------------------------------------------------------------------------
    figureWidth: float = 7.0             # in
    trueScale: bool = True               # same µm per inch in r and z; otherwise panels are panelHeight tall
    panelHeight: float = 0.9             # in, used when trueScale is False
    leftMargin: float = 0.55             # in, room for the r-axis labels
    rightMargin: float = 0.15            # in

    # ---- Acoustic overlay (pressure nodes from the run's parameters) ----------------------------------------------
    showAcousticNodes: bool = False
    nodeColor: str = "#902c32"
    nodeLineWidth: float = 0.3           # pt

    # ---- Parameter table ------------------------------------------------------------------------------------------
    showTable: bool = True
    tableTitle: str = "Simulation parameters"
    tableColumns: int = 3                # side-by-side groups of (symbol, value)
    tableLineWidth: float = 0.4          # pt
    hideAcousticRowsWhenOff: bool = True
    # (quantity in parameters.csv, symbol, format, scale applied before formatting)
    tableRows: list = field(default_factory=lambda: [
        ("forcing_frequency", "$f$", "{:.2f} kHz", 1),
        ("nozzle_radius", "$r_n$", "{:.1f} µm", 1),
        ("jet_velocity", "$v_{jet}$", "{:.2f} m/s", 1),
        ("nozzle_modulation", r"$\delta$", "{:.3f}", 1),
        ("liquid_density", r"$\rho$", "{:.0f} kg/m³", 1),
        ("surface_tension", r"$\gamma$", "{:.1f} mN/m", 1e3),
        ("viscosity", r"$\mu$", "{:.2f} mPa s", 1),
        ("ohnesorge", "Oh", "{:.4f}", 1),
        ("weber", "We", "{:.2f}", 1),
        ("cell_count", "$N$", "{:.0f}", 1),
        ("acoustic_mode", "acoustic", "{}", None),
        ("acoustic_pressure", "$p_A$", "{:.0f} Pa", 1),
        ("acoustic_frequency", "$f_{ac}$", "{:.3f} MHz", 1),
        ("acoustic_node_spacing", r"$\Delta_{node}$", "{:.1f} µm", 1),
        ("acoustic_node_velocity", "$v_{node}$", "{:.2f} m/s", 1),
    ])

    # ---- Output ---------------------------------------------------------------------------------------------------
    fileFormat: str = "png"
    dpi: int = 400


SETTINGS = Settings()

# Values of main.cpp at the time parameters.csv was introduced, for runs that predate it
MAIN_CPP_DEFAULTS = {"nozzle_radius": 18.5, "jet_velocity": 5.4, "forcing_frequency": 32.8494,
                     "nozzle_modulation": 0.005, "liquid_density": 1098, "surface_tension": 0.0679, "viscosity": 3.65,
                     "ohnesorge": 0.0983, "weber": 8.7235, "capillary_time": 10.1187, "cell_count": 1000,
                     "pinch_radius": 100}


# ---- Data -----------------------------------------------------------------------------------------------------------

class Run:
    """One results/<run> folder: jet frames, drops per frame, and the run's parameters."""

    def __init__(self, directory):
        self.directory = directory.rstrip("/")
        self.name = os.path.basename(self.directory)
        jet = np.loadtxt(f"{self.directory}/jet.csv", delimiter=",", skiprows=1, ndmin=2)
        self.frameTimes, self.jetLengths, self.radii = jet[:, 0], jet[:, 1], jet[:, 2:]
        with warnings.catch_warnings():  # a run with no drops has a header-only file
            warnings.simplefilter("ignore")
            self.drops = np.loadtxt(f"{self.directory}/drops.csv", delimiter=",", skiprows=1, ndmin=2).reshape(-1, 4)
        self.parameters, self.parametersRecorded = readParameters(self.directory)

    def frameIndex(self, time):
        return int(np.argmin(np.abs(self.frameTimes - time)))

    def jetProfile(self, frame):
        """(z, h) from the nozzle to the tip, in µm: h[0] at the nozzle, the cell centres, then 0 at the tip."""
        cellCount = self.radii.shape[1]
        length = self.jetLengths[frame]
        z = np.concatenate([[0.0], (np.arange(cellCount) + 0.5) * length / cellCount, [length]])
        h = np.concatenate([[self.radii[frame, 0]], self.radii[frame], [0.0]])
        return z, h

    def dropsInFrame(self, frame):
        """(z, R) of every drop in flight, in µm."""
        rows = self.drops[np.isclose(self.drops[:, 0], self.frameTimes[frame])]
        return rows[:, 1], rows[:, 2]

    def summary(self):
        mode = self.parameters.get("acoustic_mode", "?")
        return f"{self.name}  {self.frameTimes[-1]:.0f} µs  acoustic: {mode}"


def readParameters(directory):
    """Parameters from parameters.csv; for older runs, main.cpp defaults plus whatever acoustic.txt says."""
    parameters, recorded = {}, os.path.exists(f"{directory}/parameters.csv")
    if recorded:
        for line in open(f"{directory}/parameters.csv").read().splitlines()[1:]:
            quantity, value = line.split(",")[:2]
            parameters[quantity] = float(value)
    else:
        parameters.update(MAIN_CPP_DEFAULTS)
        parameters.update(parseAcousticSummary(f"{directory}/acoustic.txt"))
    onJet, onDrops = parameters.get("acoustic_on_jet", 0), parameters.get("acoustic_on_drops", 0)
    parameters["acoustic_mode"] = {(1, 1): "jet + drops", (1, 0): "jet", (0, 1): "drops"}.get((onJet, onDrops), "off")
    return parameters, recorded


def parseAcousticSummary(path):
    if not os.path.exists(path):
        return {}  # predates the acoustic model: no wave
    text = open(path).read()
    patterns = {"acoustic_pressure": r"pA = ([\d.]+) Pa", "acoustic_frequency": r"f = ([\d.]+) MHz",
                "acoustic_node_spacing": r"node spacing = ([\d.]+) um", "acoustic_node_velocity": r"move at ([\d.]+)",
                "acoustic_node_start": r"one at z = ([-\d.]+) um"}
    found = {key: float(m.group(1)) for key, pattern in patterns.items() if (m := re.search(pattern, text))}
    if m := re.search(r"on jet = (\w+), on drops = (\w+)", text):
        found["acoustic_on_jet"], found["acoustic_on_drops"] = float(m.group(1) == "true"), float(m.group(2) == "true")
    return found


def listRuns():
    return [d.rstrip("/") for d in sorted(glob.glob("results/*/")) if os.path.exists(f"{d}/jet.csv")]


# ---- Drawing --------------------------------------------------------------------------------------------------------

def panelTimes(run, s):
    stop = run.frameTimes[-1] if s.stopTime is None else min(s.stopTime, run.frameTimes[-1])
    start = max(run.frameTimes[0], stop - 20.0) if s.startTime is None else s.startTime
    if not s.panelize:
        return [start]
    return list(np.arange(start, stop + 1e-9, s.interval)) or [start]


def domainEnd(run, frames, s):
    if s.domainLength is not None:
        return s.domainStart + s.domainLength
    farthest = max(max((z + r for z, r in zip(*run.dropsInFrame(f))), default=0.0) for f in frames)
    return max(farthest, max(run.jetLengths[f] for f in frames)) * 1.02


def semicircle(center, radius, upper=True, points=90):
    theta = np.linspace(np.pi, 0.0, points)
    return center + radius * np.cos(theta), (1 if upper else -1) * radius * np.sin(theta)


def closedPolygon(z, h):
    """Region between the curve (z, h) and the axis, as a closed Path."""
    vertices = np.column_stack([np.concatenate([[z[0]], z, [z[-1], z[0]]]), np.concatenate([[0.0], h, [0.0, 0.0]])])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
    return Path(vertices, codes)


def compoundPath(paths):
    return Path.make_compound_path(*paths) if paths else None


def dualColorText(ax, x, y, text, fillClip, s, offset=(0, 0), ha="left", va="baseline", fontsize=None):
    """Text at data point (x, y) plus an offset in pt: textColor on the background and textOnFillColor where it lies on
    a filled shape. Drawn as glyph outlines, because the Agg renderer ignores clip paths on ordinary text."""
    glyphs = TextPath((0, 0), text, size=fontsize or s.fontSize, prop=FontProperties(family=plt.rcParams["font.family"]))
    extents = glyphs.get_extents()
    alignX = {"left": -extents.x0, "center": -(extents.x0 + extents.x1) / 2, "right": -extents.x1}[ha]
    alignY = {"baseline": 0.0, "bottom": -extents.y0, "center": -(extents.y0 + extents.y1) / 2, "top": -extents.y1}[va]
    placement = (Affine2D().translate(alignX + offset[0], alignY + offset[1]).scale(1 / 72)
                 + ax.figure.dpi_scale_trans + ScaledTranslation(x, y, ax.transData))
    ax.add_patch(PathPatch(glyphs, transform=placement, facecolor=s.textColor, edgecolor="none", zorder=5,
                           clip_on=False))
    if fillClip is not None:
        overlay = PathPatch(glyphs, transform=placement, facecolor=s.textOnFillColor, edgecolor="none", zorder=6)
        overlay.set_clip_path(fillClip)
        ax.add_patch(overlay)


def drawPanel(ax, run, frame, timeLabel, zStart, zEnd, s, showRadiusLabels):
    pointsToMicrometres = ax.figure.dpi_scale_trans.transform((1 / 72, 0))[0]  # px per pt
    pointsToMicrometres /= ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]  # µm per pt
    outline = s.outlineColor or s.fillColor
    domain = Bbox([[zStart, -10 * s.radiusLimit], [zEnd, 10 * s.radiusLimit]])
    domainClip = Rectangle((zStart, -10 * s.radiusLimit), zEnd - zStart, 20 * s.radiusLimit,
                           transform=ax.transData, visible=False)
    ax.add_patch(domainClip)

    # Shapes: the jet and each drop, as an upper filled region and a lower dashed outline
    z, h = run.jetProfile(frame)
    jetFill, satelliteFill = [closedPolygon(z, h)], []
    upperCurves, lowerCurves = [(z, h)], [(z, -h)]
    for zc, radius in zip(*run.dropsInFrame(frame)):
        upperZ, upperH = semicircle(zc, radius)
        isSatellite = s.satelliteColor is not None and radius < s.satelliteRadius
        (satelliteFill if isSatellite else jetFill).append(closedPolygon(upperZ, upperH))
        upperCurves.append((upperZ, upperH))
        lowerCurves.append(semicircle(zc, radius, upper=False))
    jetFill = [p.clip_to_bbox(domain) for p in jetFill]
    satelliteFill = [p.clip_to_bbox(domain) for p in satelliteFill]
    jetFill, satelliteFill = [p for p in jetFill if len(p.vertices)], [p for p in satelliteFill if len(p.vertices)]

    for paths, color in [(jetFill, s.fillColor), (satelliteFill, s.satelliteColor)]:
        if paths:
            ax.add_patch(PathPatch(compoundPath(paths), facecolor=color, edgecolor="none", zorder=1))
    for curves, style, width in [(upperCurves, "solid", s.outlineWidth), (lowerCurves, s.dashPattern, s.dashedWidth)]:
        for cz, ch in curves:
            line, = ax.plot(cz, ch, color=outline, lw=width, ls=style, zorder=2, solid_capstyle="round")
            line.set_clip_path(domainClip)
    allFilled = compoundPath(jetFill + satelliteFill)
    fillClip = PathPatch(allFilled, transform=ax.transData, visible=False) if allFilled is not None else None
    if fillClip is not None:
        ax.add_patch(fillClip)

    # Acoustic pressure nodes, z = z0 + v t + m spacing
    p = run.parameters
    if s.showAcousticNodes and p.get("acoustic_mode", "off") != "off" and "acoustic_node_spacing" in p:
        spacing, t = p["acoustic_node_spacing"], run.frameTimes[frame]
        first = p.get("acoustic_node_start", 0.0) + p["acoustic_node_velocity"] * t  # µm/µs = m/s
        first -= math.floor((first - zStart) / spacing) * spacing
        for zNode in np.arange(first, zEnd, spacing):
            ax.plot([zNode, zNode], [-s.radiusLimit, s.radiusLimit], color=s.nodeColor, lw=s.nodeLineWidth,
                    ls=(0, (1, 2)), zorder=0.5)

    # z axis: centre line with an arrowhead (the scale is on the ruler below the panels)
    axisStyle = dict(color=s.axisColor, lw=s.axisWidth, zorder=3, solid_capstyle="butt")
    arrowEnd = zEnd + 2 * s.arrowSize * pointsToMicrometres
    ax.annotate("", xy=(arrowEnd, 0), xytext=(zStart, 0), annotation_clip=False, zorder=3,
                arrowprops=dict(arrowstyle="-|>", lw=s.axisWidth, color=s.axisColor, shrinkA=0, shrinkB=0,
                                mutation_scale=s.arrowSize * 1.6))
    major = s.majorTickLength * pointsToMicrometres

    # r axis: labelled from 0 to +radiusLimit (render() draws its line through all panels)
    ax.plot([zStart - major, zStart], [s.radiusLimit, s.radiusLimit], **axisStyle)
    if showRadiusLabels:
        for r in (0, s.radiusLimit):
            ax.annotate(f"{r:g}", (zStart, r), xytext=(-5, 0), textcoords="offset points", ha="right", va="center",
                        fontsize=s.fontSize, color=s.textColor, annotation_clip=False)
        ax.annotate(s.radiusAxisLabel, (zStart, s.radiusLimit / 2), xytext=(-20, 0), textcoords="offset points",
                    ha="center", va="center", rotation=90, fontsize=s.fontSize, color=s.textColor,
                    annotation_clip=False)

    dualColorText(ax, zStart, s.radiusLimit, timeLabel, fillClip, s, offset=(6, 0), ha="left", va="center",
                  fontsize=s.timeLabelSize)


def rulerHeight(s):
    """in, from the ruler bar to the bottom of its labels."""
    return (s.majorTickLength + 2 + 1.3 * s.fontSize + (1.6 * s.fontSize if s.lengthAxisLabel else 0)) / 72


def drawRuler(ax, zStart, zEnd, s):
    """A bar from zStart to zEnd with ticks hanging below it and labels under the major ticks. y is in pt below the bar."""
    style = dict(color=s.axisColor, lw=s.axisWidth, solid_capstyle="butt")
    ax.plot([zStart, zEnd], [0, 0], **style)
    if s.minorTick > 0:
        for zTick in np.arange(math.ceil(zStart / s.minorTick - 1e-9) * s.minorTick, zEnd + 1e-9, s.minorTick):
            ax.plot([zTick, zTick], [0, -s.minorTickLength], **style)
    majors = np.arange(math.ceil(zStart / s.majorTick - 1e-9) * s.majorTick, zEnd + 1e-9, s.majorTick)
    for zTick in majors:
        ax.plot([zTick, zTick], [0, -s.majorTickLength], **style)
        ax.text(zTick, -s.majorTickLength - 2, f"{zTick:g}", ha="center", va="top", fontsize=s.fontSize,
                color=s.textColor)
    if s.lengthAxisLabel:
        ax.text((zStart + zEnd) / 2, -s.majorTickLength - 2 - 1.5 * s.fontSize, s.lengthAxisLabel, ha="center",
                va="top", fontsize=s.fontSize, color=s.textColor)


def tableEntries(run, s):
    p = run.parameters
    entries = []
    for quantity, symbol, form, scale in s.tableRows:
        isAcoustic = quantity.startswith("acoustic") and quantity != "acoustic_mode"
        if isAcoustic and s.hideAcousticRowsWhenOff and p.get("acoustic_mode") == "off":
            continue
        if quantity not in p:
            entries.append((symbol, "—"))
        else:
            entries.append((symbol, form.format(p[quantity] if scale is None else p[quantity] * scale)))
    return entries


def tableHeight(entryCount, s):
    rowHeight = 1.9 * s.tableFontSize / 72
    rows = math.ceil(entryCount / s.tableColumns)
    return 1.6 * rowHeight + rows * rowHeight + 0.3 * rowHeight


def drawTable(ax, run, entries, width, height, s):
    """Table in inch coordinates: a title row, then tableColumns groups of (symbol, value) rows."""
    ax.set_xlim(0, width), ax.set_ylim(0, height), ax.axis("off")
    rowHeight = 1.9 * s.tableFontSize / 72
    # Borders lie on the edge of the axes, so they must not be clipped to it
    line = dict(color=s.axisColor, lw=s.tableLineWidth, solid_capstyle="butt", clip_on=False)
    ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor=s.axisColor, lw=s.tableLineWidth,
                           clip_on=False))
    titleBottom = height - 1.6 * rowHeight
    ax.plot([0, width], [titleBottom, titleBottom], **line)
    ax.text(width / 2, height - 0.8 * rowHeight, s.tableTitle, ha="center", va="center", fontsize=s.tableFontSize + 1,
            color=s.textColor)
    rows = math.ceil(len(entries) / s.tableColumns)
    groupWidth = width / s.tableColumns
    for group in range(1, s.tableColumns):
        ax.plot([group * groupWidth] * 2, [0, titleBottom], **line)
    for i, (symbol, value) in enumerate(entries):
        group, row = divmod(i, rows)
        x, y = group * groupWidth, titleBottom - (row + 0.65) * rowHeight
        ax.text(x + 0.12, y, symbol, ha="left", va="center", fontsize=s.tableFontSize, color=s.textColor)
        ax.text(x + 0.12 + 0.32 * groupWidth, y, value, ha="left", va="center", fontsize=s.tableFontSize,
                color=s.textColor)
    if not run.parametersRecorded:
        ax.text(0, -0.6 * rowHeight, "This run predates parameters.csv: model values are main.cpp defaults; "
                "acoustic values are read from acoustic.txt.", ha="left", va="top", fontsize=s.tableFontSize - 1,
                color=s.textColor, style="italic")


def applyFonts(s):
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in s.fontFamilies if f in installed), "DejaVu Sans")
    plt.rcParams.update({"lines.scale_dashes": False, "font.family": family, "mathtext.fontset": "custom",
                         "mathtext.rm": family, "mathtext.sf": family, "mathtext.cal": family,
                         "mathtext.it": f"{family}:italic", "mathtext.bf": f"{family}:bold"})


def render(fig, run, s):
    """Clears fig and draws the panels and table for run with settings s; returns the panel times used."""
    applyFonts(s)
    fig.clear()
    fig.set_facecolor(s.background)
    times = panelTimes(run, s)
    frames = [run.frameIndex(t) for t in times]
    zStart, zEnd = s.domainStart, domainEnd(run, frames, s)

    # Layout in inches; with trueScale, µm per inch is the same in r and z
    plotWidth = s.figureWidth - s.leftMargin - s.rightMargin
    arrowRoom = 0.12  # in, beyond zEnd for the arrowhead
    micrometresPerInch = (zEnd - zStart) / (plotWidth - arrowRoom)
    yLimit = 1.12 * s.radiusLimit
    panelHeight = 2 * yLimit / micrometresPerInch if s.trueScale else s.panelHeight
    entries = tableEntries(run, s) if s.showTable else []
    tableSpace = (0.25 + tableHeight(len(entries), s) + (0.15 if not run.parametersRecorded else 0)) if s.showTable else 0
    top, bottom = 0.12, 0.12
    rulerSpace = s.rulerGap + rulerHeight(s) if s.showRuler else 0
    panelsBottom = bottom + tableSpace + rulerSpace  # in, from the bottom of the figure
    height = top + len(frames) * panelHeight + (len(frames) - 1) * s.panelGap + rulerSpace + tableSpace + bottom
    fig.set_size_inches(s.figureWidth, height, forward=True)

    origin = {"start": times[0] if s.panelize else 0.0, "absolute": 0.0}.get(s.timeOrigin, s.timeOrigin)
    panels = []
    for i, frame in enumerate(frames):
        panelTop = height - top - i * (panelHeight + s.panelGap)
        ax = fig.add_axes([s.leftMargin / s.figureWidth, (panelTop - panelHeight) / height,
                           plotWidth / s.figureWidth, panelHeight / height])
        ax.set_xlim(zStart, zStart + plotWidth * micrometresPerInch)
        ax.set_ylim(-yLimit, yLimit)
        ax.axis("off")
        panels.append(ax)
        drawPanel(ax, run, frame, s.timeLabelFormat.format(run.frameTimes[frame] - origin), zStart, zEnd, s,
                  showRadiusLabels=s.radiusLabelPanels == "all" or i == 0)

    # r-axis line: one continuous line from +radiusLimit on the top panel to -radiusLimit on the bottom one
    toFigure = fig.transFigure.inverted()
    (x, yTop), = toFigure.transform(panels[0].transData.transform([(zStart, s.radiusLimit)]))
    (_, yBottom), = toFigure.transform(panels[-1].transData.transform([(zStart, -s.radiusLimit)]))
    fig.add_artist(Line2D([x, x], [yBottom, yTop], transform=fig.transFigure, color=s.axisColor, lw=s.axisWidth,
                          solid_capstyle="butt", zorder=3))

    if s.showRuler:
        rulerBar = panelsBottom - s.rulerGap
        ax = fig.add_axes([s.leftMargin / s.figureWidth, (rulerBar - rulerHeight(s)) / height,
                           plotWidth / s.figureWidth, rulerHeight(s) / height])
        ax.set_xlim(zStart, zStart + plotWidth * micrometresPerInch)
        ax.set_ylim(-rulerHeight(s) * 72, 0)
        ax.axis("off")
        drawRuler(ax, zStart, zEnd, s)

    if s.showTable:
        tableWidth = plotWidth
        tableTop = bottom + tableSpace - 0.25
        tableHeightInches = tableHeight(len(entries), s)
        ax = fig.add_axes([s.leftMargin / s.figureWidth, (tableTop - tableHeightInches) / height,
                           tableWidth / s.figureWidth, tableHeightInches / height])
        drawTable(ax, run, entries, tableWidth, tableHeightInches, s)
    return [run.frameTimes[f] for f in frames]


def save(fig, run, times, s):
    span = f"{times[0]:.0f}-{times[-1]:.0f}us" if len(times) > 1 else f"{times[0]:.0f}us"
    path = f"{run.directory}/jetProfile_{span}.{s.fileFormat}"
    fig.savefig(path, dpi=s.dpi, facecolor=s.background)
    print(f"saved {path}")


# ---- Interactive window -------------------------------------------------------------------------------------------

class Controls:
    """A separate window of widgets that re-renders the figure, so saved figures carry no widgets."""

    def __init__(self, runs, initialRun, s):
        self.runPaths, self.settings, self.cache = runs[-12:], s, {}
        if initialRun not in self.runPaths:
            self.runPaths.append(initialRun)
        self.run = self.load(initialRun)
        self.figure = plt.figure("Jet profile")
        self.panel = plt.figure("Controls", figsize=(4.2, 6.2))
        self.panel.set_facecolor("white")

        runAxes = self.panel.add_axes([0.04, 0.55, 0.92, 0.43])
        runAxes.set_title("run", fontsize=8, loc="left")
        self.runPicker = RadioButtons(runAxes, [self.load(p).summary() for p in self.runPaths],
                                      active=self.runPaths.index(initialRun))
        for label in self.runPicker.labels:
            label.set_fontsize(7)
        self.runPicker.on_clicked(self.pickRun)

        lastTime = self.run.frameTimes[-1]
        stop = lastTime if s.stopTime is None else s.stopTime
        start = stop - 20.0 if s.startTime is None else s.startTime
        self.sliders = {}
        for row, (key, label, low, high, value, step) in enumerate([
                ("start", "start (µs)", 0.0, lastTime, start, 1.0),
                ("span", "span (µs)", 0.0, 100.0, stop - start, 1.0),
                ("interval", "interval (µs)", 1.0, 20.0, s.interval, 1.0),
                ("domainStart", "z from (µm)", 0.0, 1500.0, s.domainStart, 10.0),
                ("domainLength", "z length (µm)", 100.0, 2500.0, s.domainLength or 1000.0, 10.0)]):
            axes = self.panel.add_axes([0.3, 0.47 - row * 0.06, 0.55, 0.035])
            self.sliders[key] = Slider(axes, label, low, high, valinit=value, valstep=step)
            self.sliders[key].label.set_fontsize(7)
            self.sliders[key].on_changed(lambda _: self.redraw())
        self.toggles = CheckButtons(self.panel.add_axes([0.04, 0.06, 0.45, 0.1]), ["panels", "acoustic nodes"],
                                    [s.panelize, s.showAcousticNodes])
        self.toggles.on_clicked(lambda _: self.redraw())
        self.saveButton = Button(self.panel.add_axes([0.6, 0.07, 0.3, 0.07]), "Save")
        self.saveButton.on_clicked(lambda _: save(self.figure, self.run, self.times, self.current()))
        self.redraw()

    def load(self, path):
        if path not in self.cache:
            self.cache[path] = Run(path)
        return self.cache[path]

    def pickRun(self, label):
        self.run = self.load(self.runPaths[[self.load(p).summary() for p in self.runPaths].index(label)])
        start = self.sliders["start"]
        start.valmax = self.run.frameTimes[-1]
        start.ax.set_xlim(start.valmin, start.valmax)
        if start.val > start.valmax:
            start.set_val(start.valmax)  # triggers a redraw
        else:
            self.redraw()

    def current(self):
        v = {key: slider.val for key, slider in self.sliders.items()}
        panelize, nodes = self.toggles.get_status()
        return replace(self.settings, startTime=v["start"], stopTime=v["start"] + v["span"], interval=v["interval"],
                       domainStart=v["domainStart"], domainLength=v["domainLength"], panelize=panelize,
                       showAcousticNodes=nodes)

    def redraw(self):
        self.times = render(self.figure, self.run, self.current())
        self.figure.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="Jet profile figures from a microjet run.")
    parser.add_argument("run", nargs="?", help="results/<run> folder (default: the latest)")
    parser.add_argument("--save", action="store_true", help="render to a file in the run folder, no window")
    parser.add_argument("--start", type=float), parser.add_argument("--stop", type=float)
    parser.add_argument("--interval", type=float)
    parser.add_argument("--single", type=float, metavar="TIME", help="one frame at TIME µs instead of panels")
    parser.add_argument("--domain", type=float, nargs=2, metavar=("START", "LENGTH"), help="µm")
    parser.add_argument("--format", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    s = SETTINGS
    if args.start is not None: s = replace(s, startTime=args.start)
    if args.stop is not None: s = replace(s, stopTime=args.stop)
    if args.interval is not None: s = replace(s, interval=args.interval)
    if args.single is not None: s = replace(s, panelize=False, startTime=args.single)
    if args.domain is not None: s = replace(s, domainStart=args.domain[0], domainLength=args.domain[1])
    if args.format is not None: s = replace(s, fileFormat=args.format)

    runs = listRuns()
    runPath = (args.run or runs[-1]).rstrip("/")
    if args.save:
        run = Run(runPath)
        fig = plt.figure()
        save(fig, run, render(fig, run, s), s)
    else:
        controls = Controls(runs, runPath, s)  # noqa: F841 (widgets must stay referenced)
        plt.show()


if __name__ == "__main__":
    main()
