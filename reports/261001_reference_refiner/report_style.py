"""Drawing helpers reused from the earlier architecture deck."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT
pages, texts, source_files, fitted_texts = [], [], set(), []
import hashlib

import html

import json

import math

import subprocess

import textwrap

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle

from PIL import Image

C = dict(bg="#F2F4F5", teal="#006982", green="#00835C", orange="#D97B00",
         palegreen="#E6F2EC", paleorange="#FFF0D5", blue="#E4F0F6", ink="#20313B",
         muted="#51646E", line="#B8C7CE", white="#FFFFFF", purple="#7351A8")

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "pdf.fonttype": 42, "savefig.facecolor": C["bg"]})

def text(ax, x, y, value, size=16, color=None, weight="normal", ha="left", va="top", **kw):
    artist = ax.text(x, y, value, fontsize=size, color=color or C["ink"], weight=weight,
                     ha=ha, va=va, linespacing=1.3, **kw)
    texts.append(artist)
    return artist

def para(ax, x, y, value, width=65, size=16, **kw):
    return text(ax, x, y, "\n".join(textwrap.wrap(value, width, break_long_words=False)), size, **kw)

def rect(ax, x, y, w, h, fill="white", edge="line", radius=14, lw=1.2):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=C.get(fill, fill), edgecolor=C.get(edge, edge), linewidth=lw)
    ax.add_patch(patch)

def fit_text(artist, ax, x, y, w, h):
    """Fit rendered text to its actual card, not merely the page boundary."""
    bounds = ax.transData.transform([[x, y], [x+w, y+h]])
    available = abs(bounds[1]-bounds[0])
    bb = artist.get_window_extent(ax.figure.canvas.get_renderer())
    ratio = min(1, available[0]/max(bb.width, 1), available[1]/max(bb.height, 1))
    artist.set_fontsize(artist.get_fontsize()*ratio*.985)
    fitted_texts.append((artist, (x,y,w,h)))

def box(ax, x, y, w, h, label, fill="green", size=17, badge=None):
    rect(ax, x, y, w, h, fill=fill, edge=fill)
    color = C["white"] if fill in ("green", "teal", "orange", "purple") else C["ink"]
    label = "\n".join("\n".join(textwrap.wrap(line, max(12, int((w-28)/(size*.67))), break_long_words=False)) for line in label.split("\n"))
    artist = text(ax, x+w/2, y+h/2, label, size, color, "bold", ha="center", va="center")
    fit_text(artist, ax, x+12, y+10, w-24, h-20)
    if badge:
        ax.add_patch(Circle((x+10, y+8), 19, facecolor=C["orange"], edgecolor="white", linewidth=2))
        text(ax, x+10, y+8, badge, 14, "white", "bold", ha="center", va="center")

def arrow(ax, points, color="green", dashed=False):
    color = C.get(color, color)
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=2, ls="--" if dashed else "-", zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=14,
                                linewidth=2, color=color, linestyle="--" if dashed else "-", zorder=1))

def card(ax, x, y, w, h, title, body, accent="teal", size=16):
    rect(ax, x, y, w, h)
    heading=text(ax, x+22, y+18, title, 18, C[accent], "bold")
    fit_text(heading,ax,x+22,y+18,w-44,32)
    artist=para(ax, x+22, y+60, body, width=int((w-44)/(size*.72)), size=size)
    fit_text(artist,ax,x+22,y+60,w-44,h-78)

def banner(ax, y, body, fill="blue", size=16):
    box(ax, 45, y, 1510, 62, body, fill=fill, size=size)

def page(title, subtitle, source):
    fig = plt.figure(figsize=(16, 9), dpi=110, facecolor=C["bg"])
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1600), ylim=(900, 0))
    ax.axis("off")
    rect(ax, 15, 12, 1570, 77, fill="teal", edge="teal", radius=0)
    text(ax, 44, 50, title, 25, "white", "bold", va="center")
    text(ax, 45, 106, subtitle, 13, C["muted"])
    text(ax, 45, 867, source, 9, C["muted"])
    text(ax, 1550, 867, str(len(pages)+1), 10, C["muted"], ha="right")
    pages.append((fig, title))
    return fig, ax

def snippet(ax, x, y, w, h, path, first, last, size=12):
    source_files.add(path)
    raw = path.read_text().splitlines()[first-1:last]
    code = textwrap.dedent("\n".join(raw)).splitlines()
    columns = max(40, int((w-36)*.72/(size*.602))-5)
    displayed=[]
    for i,line in enumerate(code):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent=len(line)-len(line.lstrip())
        wrapped=textwrap.wrap(line.lstrip(),columns,initial_indent=" "*indent,subsequent_indent=" "*(indent+4),
                              break_long_words=False,break_on_hyphens=False,
                              replace_whitespace=False,drop_whitespace=True)
        displayed.extend((f"{first+i:3d}  " if j==0 else "  ↳  ")+part for j,part in enumerate(wrapped))
    label = (str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else "clean_new/"+str(path.relative_to(OLD)))
    label = label.replace("sources/ai-toolkit-flux/extensions_built_in/diffusion_models/flux2/src/", "Toolkit/flux2/src/")
    rect(ax, x, y, w, h, fill="#EAF0F3", edge="line", radius=8)
    text(ax, x+17, y+12, f"{label}  L{first}–{last}", 10, C["teal"], "bold")
    size = min(size, (w-36)*.72/max(map(len, displayed))/.602, (h-55)*.72/(len(displayed)*1.3))
    numbered = "\n".join(displayed)
    text(ax, x+17, y+43, numbered, size, family="DejaVu Sans Mono")

def image_panel(fig, path, x, y, w, h):
    ax = fig.add_axes([x/1600, 1-(y+h)/900, w/1600, h/900])
    ax.imshow(Image.open(path)); ax.axis("off")

def table(ax, x, y, widths, headers, rows, row_h=67, size=13):
    heights = [48] + [row_h]*len(rows)
    for ri, values in enumerate([headers]+rows):
        xx = x
        for ci, (w, value) in enumerate(zip(widths, values)):
            fill = C["teal"] if ri == 0 else (C["white"] if ri % 2 else "#E8F0F2")
            ax.add_patch(Rectangle((xx,y),w,heights[ri],facecolor=fill,edgecolor=C["line"],lw=.8))
            lines = "\n".join(textwrap.wrap(value, int((w-24)/(size*.80)), break_long_words=False))
            artist=text(ax, xx+12, y+heights[ri]/2, lines, size, "white" if ri==0 else C["ink"],
                        "bold" if ri==0 or ci==0 else "normal", va="center")
            fit_text(artist,ax,xx+12,y+7,w-24,heights[ri]-14)
            xx += w
        y += heights[ri]
