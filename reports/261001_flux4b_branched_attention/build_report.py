"""Build the current FLUX architecture report as vector PDF, SVG pages and HTML.

Run from the repository root with envs/report/bin/python. Matplotlib and Pillow
are required. Source excerpts are read from the audited local checkouts.
"""

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

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = Path("/home/kolyangg/rsrch_clean_new/diffusion_template")
MODEL = ROOT / "sources/ai-toolkit-flux/extensions_built_in/diffusion_models/flux2/src/model.py"
CORE = ROOT / "ba_dit/nn/reference_read_delta.py"
SEAM = ROOT / "ba_dit/backends/attention.py"
RUNTIME = ROOT / "ba_dit/backends/flux_runtime.py"
REGISTER = ROOT / "ba_dit/backends/flux2_native.py"
LEGACY = OLD / "src/model/photomaker_branched/hardcase_attn_processor.py"
MANIFEST = ROOT / "runs/flux48_one_id_diagnostic/checkpoint-001500/manifest.json"
C = dict(bg="#F2F4F5", teal="#006982", green="#00835C", orange="#D97B00",
         palegreen="#E6F2EC", paleorange="#FFF0D5", blue="#E4F0F6", ink="#20313B",
         muted="#51646E", line="#B8C7CE", white="#FFFFFF", purple="#7351A8")
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "pdf.fonttype": 42, "savefig.facecolor": C["bg"]})
pages, texts, source_files, fitted_texts = [], [], set(), []


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


# 1 — scope and the exact run, without importing/loading model weights.
fig, ax = page("FLUX.2-klein Base 4B — the architecture we train",
               "Architecture audit · 1 October 2026 · current one-ID run · compared with the original CL39 PhotoMaker system",
               "Sources: configs/flux4b_48_one_id.yaml; checkpoint-001500/manifest.json; ba_dit/adapters.py; evidence through 08:41 UTC.")
card(ax,45,157,475,210,"BACKBONE", "Base / undistilled FLUX.2-klein 4B. Frozen BF16 transformer: 5 double-stream blocks + 20 single-stream blocks. Width 3,072; 24 heads of width 128.")
card(ax,562,157,475,210,"TRAINABLE BRANCH", "Eight sites. Separate rank-16 K and V deltas at each site. 32 FP32 A/B tensors; 1,572,864 parameters. Native Q/K/V and output weights stay frozen.","orange")
card(ax,1079,157,476,210,"CURRENT EXPERIMENT", "flux48_one_id_diagnostic; branch_only. Nineteen training views, twelve validation prompts, 768 px targets and 512 px reference budget. One RTX 6000 Ada.")
text(ax,800,427,r"$A_{\mathrm{out}} = A_{\mathrm{joint}} + \mathrm{scatter}_{T}\,[\,0.1\,(R_1-R_0)\,]$",29,ha="center")
para(ax,180,505,"The added term measures what learned reference K/V updates change relative to an unadapted read of the same face tokens. The native joint attention already sees the full reference image.",100,19)
card(ax,45,620,735,181,"CONNECTION THAT MATTERS", "The correction is added before FLUX’s native output projection. All target-image queries receive it; text and reference rows receive zero direct correction.","orange")
card(ax,820,620,735,181,"RELATION TO CL39", "This is a CL39-inspired reference-read delta. The current implementation has no CL39 target-face router, Gaussian bands, entropy confidence, or surface-ownership loss.")

# 2 — complete context.
fig, ax = page("Whole model — text, target and reference enter one FLUX transformer",
               "The reference is appended along the token sequence; there is no PhotoMaker-style doubled target/reference batch.",
               "Sources: flux_runtime.py L32–87, L109–133; native model.py::Flux2.forward; pipeline.py::_get_qwen_prompt_embeds.")
box(ax,45,163,220,105,"Prompt text",fill="blue")
box(ax,305,163,270,105,"Frozen Qwen3-4B\n512 × 7,680 features",fill="green",size=16)
box(ax,45,333,220,110,"Target latent x_t\ntrain: VAE + noise\ninfer: seeded noise",fill="blue",size=15)
box(ax,305,333,270,110,"Pack target latent\n2,304 × 128 tokens",fill="green",size=16)
box(ax,45,515,220,110,"Reference image\n512 × 512 here",fill="blue",size=16)
box(ax,305,515,270,110,"Frozen VAE + packing\n1,024 × 128 tokens",fill="green",size=16)
for y in [215,388,570]: arrow(ax,[(265,y),(305,y)])
rect(ax,625,149,455,551,fill="palegreen",edge="green")
text(ax,852,168,"Frozen FLUX + trainable BA",18,C["green"],"bold",ha="center")
box(ax,655,211,395,76,"txt_in / img_in → width 3,072",size=16)
box(ax,655,319,395,91,"5 DOUBLE blocks\ntext/image streams\nwith joint attention",size=15)
box(ax,655,440,395,60,"Concatenate [text | target | reference]",fill="blue",size=14)
box(ax,655,530,395,92,"20 SINGLE blocks\njoint attention + parallel MLP",size=15)
text(ax,852,649,"BA at 4 double + 4 single sites",16,C["orange"],"bold",ha="center")
for a,b in [(287,319),(410,440),(500,530)]: arrow(ax,[(852,a),(852,b)])
arrow(ax,[(575,215),(655,245)])
arrow(ax,[(575,388),(606,388),(606,257),(655,257)])
arrow(ax,[(575,570),(595,570),(595,271),(655,271)])
box(ax,1130,235,420,115,"Native final norm + projection\n128-channel prediction\nKeep target rows only",size=16)
box(ax,1130,403,420,115,"50 Euler denoising steps\nCFG = 4: positive + negative calls\nBA active in both calls",fill="blue",size=16)
box(ax,1130,575,420,105,"Frozen VAE decode\n768 × 768 generated image",size=17)
arrow(ax,[(1080,575),(1102,575),(1102,293),(1130,293)])
arrow(ax,[(1340,350),(1340,403)]); arrow(ax,[(1340,518),(1340,575)])
banner(ax,745,"Timestep → native modulation / residual gates. Position IDs → native Q/K RoPE. All of these stay in the forward path.",size=15)
text(ax,48,823,"Qwen3-4B here is FLUX’s text encoder. Qwen-Image-2.1 is a different visual backbone and is not the current training run.",12,C["muted"])

# 3 — block insertion sites.
fig, ax = page("Where BA is connected — eight registered sites, zero-based indices",
               "Green = frozen native block; orange = a native block with an added ReferenceReadDelta module.",
               "Sources: ba_dit/backends/flux2_native.py::SITES; native model.py::DoubleStreamBlock.forward / SingleStreamBlock.forward.")
text(ax,50,155,"DOUBLE STREAM\n5 blocks",18,C["teal"],"bold")
for i in range(5):
    box(ax,390+i*220,149,165,66,f"D{i}",fill="orange" if i in [1,2,3,4] else "green")
    if i<4: arrow(ax,[(555+i*220,182),(610+i*220,182)])
text(ax,50,269,"SINGLE STREAM  ·  20 blocks",18,C["teal"],"bold")
for i in range(20):
    x=45+i*76
    box(ax,x,321,64,62,f"{i}",fill="orange" if i in [3,8,13,18] else "green",size=15)
    if i<19: arrow(ax,[(x+64,352),(x+76,352)])
card(ax,45,440,735,291,"DOUBLE: correct the image-attention result", "Image Q/K/V and text Q/K/V are projected separately, then attend jointly. Split the result into text and image rows. Add the target-only BA delta to img_attn, then apply img_attn.proj and the native modulation gate. The image MLP remains unchanged.",size=16)
card(ax,820,440,735,291,"SINGLE: correct only the attention part", "linear1 produces Q/K/V plus a parallel MLP input. Add the BA delta to the native attention result. Concatenate attention with the activated MLP, then apply the existing linear2 and native residual gate. BA never changes the MLP slice directly.",size=16)
banner(ax,770,"Each site: 2 low-rank projections × (A + B) = 196,608 parameters. Eight sites = 1,572,864 trainable parameters.",size=16)

# 4 — main branch scheme, exportable on its own.
fig, ax = page("Branched attention — paired reference reads, then an additive correction",
               "A single selected FLUX block · native target Q · identical gathered reference support in R0 and R1 · correction before output projection",
               "Sources: ba_dit/nn/reference_read_delta.py L47–97; ba_dit/backends/attention.py::flux_reference_delta. Schematic omits head reshape.")
box(ax,45,165,260,108,"All text + target +\nreference states",badge="A",size=17)
box(ax,360,165,365,108,"Native joint attention\nA_joint over all K/V",badge="B",size=18)
box(ax,835,165,300,108,"A_joint\n+ scatter_T(Δ)",badge="G",size=17)
box(ax,1210,165,340,108,"Native output projection\n+ gate + residual",badge="H",size=17)
arrow(ax,[(305,219),(360,219)]); arrow(ax,[(725,219),(835,219)]); arrow(ax,[(1135,219),(1210,219)])
box(ax,45,355,260,106,"Full target Q\nnative norm + RoPE",fill="blue",badge="C",size=17)
box(ax,45,551,260,124,"Reference face indices J\ngather H_R, K_pre, V\nkeep position IDs",fill="blue",badge="D",size=15)
box(ax,367,365,345,103,"K0 = RoPE(norm(K_pre))\nV0 = V",fill="palegreen",size=16)
box(ax,367,551,345,124,"ΔK = B_K A_K H_R\nΔV = B_V A_V H_R\ncarry original K_pre, V",fill="paleorange",badge="E",size=16)
box(ax,771,355,360,115,"R0 = SDPA(Q_T, K0, V0)\nunadapted reference read",fill="green",size=16)
box(ax,771,550,360,125,"R1 = SDPA(Q_T, K1, V1)\nK1 = RoPE(norm(K_pre + ΔK))\nV1 = V + ΔV",fill="orange",size=15)
box(ax,1207,430,343,123,"Δ = 0.1 × (R1 − R0)\nall target-image queries\nzero direct text/ref delta",fill="paleorange",badge="F",size=17)
arrow(ax,[(305,603),(337,603),(337,416),(367,416)])
arrow(ax,[(305,615),(367,615)])
arrow(ax,[(712,416),(771,416)])
arrow(ax,[(712,610),(771,610)],color="orange")
arrow(ax,[(305,401),(324,401),(324,308),(949,308),(949,355)],color="teal")
arrow(ax,[(324,308),(747,308),(747,510),(949,510),(949,550)],color="teal")
arrow(ax,[(1131,412),(1161,412),(1161,459),(1207,459)])
arrow(ax,[(1131,612),(1180,612),(1180,529),(1207,529)],color="orange")
arrow(ax,[(1378,430),(1378,322),(985,322),(985,273)],color="orange")
banner(ax,726,"R0 is a separate face-only SDPA read. It is NOT A_joint and is NOT the reference slice of the joint softmax output.",size=16)
text(ax,50,819,"At initialization R1 = R0, so Δ = 0. After training, reference K/V updates change the additive correction; the full native path still runs.",13,C["muted"])

# 5 — tokens and actual masks.
fig, ax = page("Token coordinates and masks — the current one-ID validation reference",
               "The native transformer keeps every reference token. Only the extra branch gathers face tokens.",
               "Sources: native model.py L169–183; attention.py L8–27; geometry.py::reference_geometry; mask_review_20261001/one_id_flux/audit.json.")
box(ax,45,151,235,62,"Text: 512 slots",fill="blue",size=16)
box(ax,280,151,920,62,"Target: 2,304 tokens = 48 × 48  |  receives BA correction everywhere",fill="paleorange",size=16)
box(ax,1200,151,355,62,"Reference: 1,024 = 32 × 32",fill="palegreen",size=14)
text(ax,50,236,"Double image stream: T = [0, 2304), R = 2304 + J.  Single stream: T = [512, 2816), R = 2816 + J.",15)
image_panel(fig,ROOT/"runs/mask_review_20261001/one_id_flux/reference_masks_1.png",45,290,1510,420)
card(ax,45,729,735,108,"REFERENCE MASK", "88 selected keys here; cap 512. Gathered positions keep their original RoPE coordinates.",size=14)
card(ax,820,729,735,108,"TARGET MASK", "No target_gate is passed. Generated-face masks are scoring-only; they never gate inference.","orange",size=14)

# 6 — mathematical contract.
fig, ax = page("The branch equation — what cancels and what remains trainable",
               "Q is reused; K adaptation is inserted before key normalization and original-coordinate RoPE; V adaptation is additive.",
               "Sources: reference_read_delta.py; attention.py; adapters.py. R0 and R1 below are conditioned on the same current hidden states.")
rect(ax,45,152,950,367,fill="white")
equations = [r"$\delta K=(\alpha/r)B_K A_K H_R,\quad \delta V=(\alpha/r)B_V A_V H_R$",
 r"$K_0=\mathrm{RoPE}_R(\mathrm{Norm}_K(K_R)),\quad K_1=\mathrm{RoPE}_R(\mathrm{Norm}_K(K_R+\delta K))$",
 r"$R_0=\mathrm{SDPA}(Q_T,K_0,V_R),\quad R_1=\mathrm{SDPA}(Q_T,K_1,V_R+\delta V)$",
 r"$\Delta=\gamma(R_1-R_0),\qquad \gamma=0.1,\quad r=\alpha=16$"]
for y,e in zip([183,271,359,447],equations): text(ax,70,y,e,20)
card(ax,1035,152,520,367,"ZERO INITIALIZATION", "A is Kaiming initialized; B is zero. Thus ΔK = ΔV = 0 and the paired reads cancel. First-step B gradients can be nonzero. A gradients start at zero because B = 0, then become active after B updates.","orange",size=17)
card(ax,45,561,475,244,"WHAT IS FROZEN", "Native model parameters, Q projection, K normalization, RoPE, output projection, native modulation and MLP weights. Freezing parameters does not detach hidden states.",size=16)
card(ax,562,561,475,244,"WHAT IS DIFFERENTIABLE", "Both reads stay in the graph. Later frozen blocks transmit gradients to earlier adapters. FLUX reference states evolve through joint attention and depend on the target.",size=16)
card(ax,1079,561,476,244,"WHAT BA-OFF MEANS", "Skip the added correction while keeping the same native reference-conditioned FLUX computation. It does not mean text-to-image without a reference.","orange",size=16)

# 7 — registration and actual input connection.
fig, ax = page("Code walkthrough 1 — register adapters and pass the reference contract",
               "Original source line numbers; comments / blank lines omitted; long lines wrapped with ↳. Native LoRA is inactive in this run.",
               "Sources printed with line numbers below. Full hashes and checkpoint identity are in source_audit.json.")
snippet(ax,45,155,700,164,REGISTER,8,11,13)
snippet(ax,45,342,700,310,REGISTER,28,39,12)
snippet(ax,785,155,770,370,RUNTIME,75,87,12)
card(ax,785,551,770,236,"HOW THIS REACHES ATTENTION", "The patched Flux2.forward turns the mask into selected reference indices, validates the target/reference split and passes a FluxBranchContext to every block. Only registered blocks consume it. The checkpoint owns all 32 A/B tensors.",size=16)
para(ax,65,691,"The full image sequence stays [target | reference]. Passing branch=False removes only the branch context; the reference tokens remain in native attention.",48,16)

# 8 — two exact seams.
fig, ax = page("Code walkthrough 2 — the exact double-stream and single-stream seams",
               "The existing attention result is preserved. The added tensor is passed through the existing projection once.",
               "Source: patched AI Toolkit flux2/src/model.py. The code is protected by the repository source-pin and patch-hash checks.")
text(ax,45,151,"DOUBLE STREAM",19,C["teal"],"bold")
snippet(ax,45,189,735,357,MODEL,468,485,12)
text(ax,820,151,"SINGLE STREAM",19,C["teal"],"bold")
snippet(ax,820,189,735,357,MODEL,366,377,12)
card(ax,45,587,735,208,"DOUBLE OUTPUT", "The branch supplies zero correction on reference rows. img_attn.proj acts on the native result plus the correction, followed by the original gate/residual. Text attention is unchanged directly at this site.",size=16)
card(ax,820,587,735,208,"SINGLE OUTPUT", "linear2 receives [attention + correction | MLP activation]. The branch therefore uses only its attention input slice. Text and reference rows have zero direct added tensor; later joint blocks can propagate effects.",size=16)

# 9 — norm and RoPE.
fig, ax = page("Code walkthrough 3 — gather reference support without changing coordinates",
               "The branch reads the same modulated reference hidden states that produced native image Q/K/V at this block.",
               "Source: ba_dit/backends/attention.py::flux_reference_delta. V is native projected value; only K receives norm and RoPE.")
snippet(ax,45,154,970,625,SEAM,6,27,14)
card(ax,1055,154,500,192,"OFFSET", "Double blocks use image-only indices. Single blocks add the text prefix length to target and reference indices.",size=16)
card(ax,1055,370,500,191,"POSITION", "Gather pe at the original reference locations. Do not renumber cropped keys or reuse target positions for reference K.","orange",size=16)
card(ax,1055,586,500,193,"QUERY SCOPE", "All target-image rows are selected. The call omits target_gate, so there is no generated-face spatial router.","orange",size=16)

# 10 — the branch implementation.
fig, ax = page("Code walkthrough 4 — learned K/V, paired SDPA, and target-only scatter",
               "The same Q, reference indices and postprocessing are used for both reads. Query chunks bound temporary attention work.",
               "Source: ba_dit/nn/reference_read_delta.py. Excerpts are literal; native in the code is the unadapted face-only R0 read.")
snippet(ax,45,153,1510,145,CORE,13,16,14)
text(ax,65,312,"Forward: two FP32 linear maps through A then B, scaled by alpha/r, and cast back to the activation dtype.",14)
snippet(ax,45,347,1510,435,CORE,78,97,13)
text(ax,50,814,"gamma is a fixed Python scalar. target_gate exists as an optional API argument but neither current backbone adapter supplies it.",15,C["orange"],"bold")

# 11 — side by side architecture.
fig, ax = page("CL39 versus current FLUX — two different residual definitions",
               "Both keep a native path and reuse target queries. Their reference support, subtraction baseline and routing are different.",
               "Sources: supplied report pp. 7, 9–14; clean_new/hardcase_attn_processor.py L428–700; current reference_read_delta.py.")
box(ax,45,153,735,62,"CL39 / PhotoMaker + SDXL U-Net",fill="teal",size=19)
box(ax,820,153,735,62,"Current / FLUX Base 4B",fill="orange",size=19)
box(ax,75,251,310,103,"N = W_o · Attn(Q_T,K_T,V_T)\ntarget self-attention",size=14)
box(ax,439,251,310,103,"R = W_o · Attn(Q_T,K_R,V_R)\nmasked reference features",size=14)
box(ax,143,413,541,143,"D = R − N\nGaussian split D = L + H\nΔ = S · C · (g_L L + g_H H)",fill="paleorange",size=19)
arrow(ax,[(231,354),(231,413)]); arrow(ax,[(594,354),(594,413)])
box(ax,165,608,495,68,"Target output = N + Δ",size=19)
arrow(ax,[(413,556),(413,608)])
box(ax,850,251,310,103,"R0: original reference K/V\nface positions gathered",size=16)
box(ax,1214,251,310,103,"R1: K/V + learned deltas\nsame gathered positions",fill="orange",size=16)
box(ax,918,413,541,143,"Δ = 0.1 · (R1 − R0)\nno Gaussian / confidence / S\nall target-image queries",fill="paleorange",size=19)
arrow(ax,[(1005,354),(1005,413)]); arrow(ax,[(1369,354),(1369,413)],color="orange")
box(ax,921,608,535,68,"Target attention = A_joint,T + Δ",size=18)
arrow(ax,[(1188,556),(1188,608)],color="orange")
para(ax,75,718,"CL39 routes finished, projected messages within a target-face region. C is applied in up_blocks.0/1; C = 1 in other routed groups.",59,15)
para(ax,850,718,"FLUX adds a pre-projection reference-adaptation delta. Full joint attention already carries native text, target and reference conditioning.",59,15)

# 12 — concise differences table.
fig, ax = page("What was carried over — and which CL39 mechanisms are absent",
               "These are implementation differences, not evidence that one architecture is better. The datasets/backbones also differ.",
               "Sources: original CL39 source/config and prior architecture report; current backend, branch module, geometry and training config.")
rows = [
 ("Backbone / reference path","SDXL U-Net + PhotoMaker; target/reference batch halves; native row-wise text/ID cross-attention.","FLUX DiT; target/reference tokens share the image stream; text joins native multimodal attention."),
 ("Residual anchor","Reference-conditioned message minus target-native message: R − N.","Adapted minus unadapted reference read: R1 − R0. Native joint result is retained."),
 ("Trainable projections","Rank-128 target/reference Q/K/V branches, with additional PM/generic trainables in the CL39 recipe.","Only rank-16 K/V delta A/B matrices at eight sites; native Q and output projections frozen."),
 ("Reference support","Zero features outside face box; excluded positions remain in softmax as zero sinks.","Gather valid face token positions, at most 512. No outside-mask branch keys. Native attention still sees all refs."),
 ("Spatial output scope","Soft target-face router S, including transition rings.","All target-image queries; no target gate supplied. Generated-face masks are evaluation-only."),
 ("Frequency / confidence","Gaussian 5×5 low/high split; timestep gains; detached entropy C in up_blocks.0/1.","Fixed gamma 0.1. No low/high bands, scheduler-dependent BA gains, or entropy confidence."),
 ("Projection / coordinates","Both candidate messages projected by W_o before routing; U-Net spatial grids.","Add delta before native projection. Preserve FLUX Q/K normalization and reference RoPE."),
 ("Objective / initialization","Inherited diffusion + CL27 surface/ownership training terms; PM-effective initialization.","Full-image native velocity MSE only. Zero B gives zero added correction at initialization."),
]
table(ax,45,149,[255,627,628],["Mechanism","CL39 predecessor","Current FLUX branch_only"],rows,row_h=76,size=13)

# 13 — explicit old code and practical implications.
fig, ax = page("CL39 code reference — spatial routing, frequency bands and confidence",
               "These controls are present in the predecessor. The current FLUX correction deliberately implements a smaller mechanism.",
               "Source: /home/kolyangg/rsrch_clean_new/diffusion_template/src/model/photomaker_branched/hardcase_attn_processor.py.")
snippet(ax,45,154,930,366,LEGACY,628,655,11.7)
snippet(ax,45,543,930,121,LEGACY,685,686,13)
card(ax,1015,154,540,290,"CL39 CONFIDENCE", "Reference-attention entropy E maps to u = sigmoid((E − 0.75)/0.08), then C = clip(1 − 0.75u, 0.25, 1). This detached, parameter-free multiplier attenuates the correction. There is no literal extra null key.",size=16)
card(ax,1015,473,540,324,"IMPLICATION FOR THIS EXPERIMENT", "The new run checks whether a compact reference-read adaptation can learn. It does not yet test CL39’s target-localized abstention or frequency control. Changes to pose/background are possible because every target query can receive the delta and later blocks mix streams.","orange",size=16)
para(ax,65,713,"A closer CL39 transfer would require a separately specified target router, confidence definition on valid reference keys, and frequency handling on the target grid. Those would be new experiments, not descriptions of the current checkpoint.",77,16)

# 14 — training and native validation pathway.
fig, ax = page("Training and validation — the same patched backend and adapter weights",
               "The diagnostic uses native full-image flow loss. It has no face-weighted loss, identity loss or CL27 surface loss.",
               "Sources: training.py L44–127; flux_runtime.py L90–125; inference.py L26–95; checkpoint.py::load_adapters / restore_training.")
for x,w,label in [(45,295,"Frozen text + VAE\nprecompute conditioning"),(403,338,"Cached pairs + FLUX\nforward / loss / backward"),(804,325,"Save adapter + optimizer\nRNG + data cursor"),(1192,363,"Fresh validation worker\nload exact checkpoint")]:
    box(ax,x,159,w,104,label,size=16)
for a,b in [(340,403),(741,804),(1129,1192)]: arrow(ax,[(a,211),(b,211)])
rect(ax,45,310,955,201,fill="white")
text(ax,75,337,r"$x_t=(1-\sigma)x_0+\sigma\epsilon,\qquad v^*=\epsilon-x_0$",24)
text(ax,75,413,r"$\mathcal{L}=\mathrm{mean}\,[\,(v_\theta(x_t,\sigma,c,\mathrm{ref})-v^*)^2\,]$",24)
text(ax,75,477,"Toolkit sigmoid timestep schedule; new noise/timestep draws for each training sample.",12,C["muted"])
card(ax,1040,310,515,201,"EFFECTIVE BATCH = 8", "Microbatch 1 × accumulation 8. AdamW, LR 1e−4, 100-update warmup, weight decay 0, gradient clip 1.0.",size=16)
card(ax,45,552,475,251,"MEMORY", "Encoders and VAE are absent from optimization. Frozen BF16 backbone remains differentiable for adapter gradients; non-reentrant checkpointing is enabled. Measured peak reserved: 9.527 GiB.",size=16)
card(ax,562,552,475,251,"VALIDATION", "Same registered sites and branch module. Strict checkpoint identity, inventory and dtype checks. Twelve fixed prompts; 50 Euler steps, CFG 4. Validation is serial at 0/500/1000/1500/2000.",size=16)
card(ax,1079,552,476,251,"INTERPRETING LOSS", "Different images, noise and timesteps make raw step loss noisy. Finite gradients, parameter updates and matched validation changes establish a live path; fixed-input probes remain useful for deeper diagnosis.","orange",size=16)

# 15 — measured evidence and limits.
fig, ax = page("Measured evidence — branches update and validation images change",
               "One-ID diagnostic only · step-1,500 evidence snapshot · final step-2,000 validation was pending at the last 08:41 UTC check",
               "Sources: runs/review_one_id/validation_change_*.json; ONE_ID_DIAGNOSTIC.md; checkpoint-001500; Comet 25604bb4dc58429d9378efea58b43ed2.")
reports=[json.loads((ROOT/f"runs/review_one_id/validation_change_000000_to_{s:06d}.json").read_text()) for s in [500,1000,1500]]
for pos,key,group,title in [(45,"id_sim","legacy_metrics","Identity similarity"),(560,"topiq_face_mean","face_quality_metrics","TOPIQ-Face")]:
    aa=fig.add_axes([pos/1600,1-470/900,465/1600,280/900])
    vals=[reports[0][group][key]["step_0"]]+[r[group][key]["later"] for r in reports]
    aa.plot([0,500,1000,1500],vals,"o-",color=C["teal"],lw=2.5)
    aa.set_title(title,fontsize=17);aa.set_xticks([0,500,1000,1500]);aa.set_xlabel("Optimizer step",fontsize=12)
    aa.tick_params(labelsize=11);aa.grid(alpha=.22);aa.set_facecolor("white")
    for s,v in zip([0,500,1000,1500],vals): aa.annotate(f"{v:.3f}",(s,v),xytext=(0,9),textcoords="offset points",ha="center",fontsize=11)
    aa.margins(y=.35)
card(ax,1090,154,465,334,"OBSERVED", "All 12 images changed at each reviewed checkpoint. At step 1,500, all 16 B matrices are nonzero; combined L2 norm 10.4308. B changed by L2 3.8779 since step 1,000. Recent gradients remain finite.","orange",size=16)
table(ax,45,549,[480,255,255,520],["Step-1,500 check","Baseline","Step 1,500","Interpretation"],[
 ("Mask-matched identity", "0.310866", "0.338955", "Improved on the same twelve prompts"),
 ("CLIP text similarity", "28.510215", "28.673897", "Small mean increase"),
 ("TOPIQ-Face", "0.687094", "0.713561", "Mean face quality increased"),
],row_h=57,size=14)
banner(ax,797,"This supports branch influence and learning on one identity. It does not establish generalization or superiority over CL39.",size=15)

# 16 — source ledger and audit.
fig, ax = page("Source ledger — exact implementation, checkpoint and reproducible figures",
               "Local code/config were checked against the step-1,500 checkpoint identity before this report was built.",
               "Report generator: reports/261001_flux4b_branched_attention/build_report.py. Assets: PDF, per-page SVG/PNG, HTML and source_audit.json.")
table(ax,45,150,[420,1090],["Source / role","Audited location or identity"],[
 ("Experiment", "configs/flux4b_48_one_id.yaml · mode=branch_only · flux48_one_id_diagnostic"),
 ("Backbone weights", "black-forest-labs/FLUX.2-klein-base-4B @ a3b4f4849157f664bdbc776fd7453c2783562f4d"),
 ("Native source", "ostris/ai-toolkit @ ecee894ed2b1f3716d9d7326693061ec1a3105bb + recorded FLUX patch"),
 ("Core / connection", "ba_dit/nn/reference_read_delta.py; backends/attention.py; backends/flux2_native.py; backends/flux_runtime.py"),
 ("Checkpoint", "checkpoint-001500/manifest.json SHA256: 87b450984641b26c845591d10b3c568438fcf4e211073a45f0c2408f782ef934"),
 ("Earlier implementation", "rsrch_clean_new: hardcase_attn_processor.py + CL39 config; base HEAD 728fd28f7c2c73127fa300cd04a62b29a96be918; current file hashes recorded"),
 ("Prior report / style", "User-supplied 15-page E13 → CL39 PDF; rsrch_apr_test/diffusion_template/tools/comet report pages and tools/analysis/build_cl39_r2_architecture_deck.js"),
 ("Verified identity", "Adapter identity, training-code digest and resolved config digest all matched checkpoint-001500. 32 tensors / 1,572,864 parameters."),
],row_h=65,size=12.8)
banner(ax,777,"Scope: the active FLUX 4B architecture. Qwen-Image and FLUX 9B are separate profiles; this report makes no trained-result claim for them.",size=14)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/"flux4b_branched_attention_architecture.pdf"
    with PdfPages(target,metadata={"Title":"Current FLUX Base 4B branched attention architecture", "Author":"rsrch_new", "Subject":"Audited architecture and comparison with CL39"}) as pdf:
        for i,(fig,title) in enumerate(pages,1):
            fig.canvas.draw()
            for artist,(x,y,w,h) in fitted_texts:
                if artist.figure is not fig: continue
                for _ in range(3):
                    bb=artist.get_window_extent(fig.canvas.get_renderer())
                    a,b=artist.axes.transData.transform([[x,y],[x+w,y+h]])
                    ratio=min((b[0]-a[0])/bb.width,(a[1]-b[1])/bb.height,1)
                    if ratio>=1: break
                    artist.set_fontsize(artist.get_fontsize()*ratio*.96)
            fig.canvas.draw()
            # Check real rendered text bounds before export. Tiny footer tolerance only.
            renderer=fig.canvas.get_renderer()
            for artist in texts:
                if artist.figure is not fig: continue
                bb=artist.get_window_extent(renderer)
                if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.width+1 or bb.y1 > fig.bbox.height+1:
                    raise ValueError(f"Page {i}: text outside canvas: {artist.get_text()[:100]}")
            for artist,(x,y,w,h) in fitted_texts:
                if artist.figure is not fig: continue
                bb=artist.get_window_extent(renderer)
                a,b=artist.axes.transData.transform([[x,y],[x+w,y+h]])
                if bb.x0<a[0]-1 or bb.x1>b[0]+1 or bb.y0<b[1]-1 or bb.y1>a[1]+1:
                    raise ValueError(f"Page {i}: text outside its card: {artist.get_text()[:100]}")
            pdf.savefig(fig)
            fig.savefig(OUT/f"page_{i:02d}.svg")
            fig.savefig(OUT/f"page_{i:02d}.png",dpi=135)
            plt.close(fig)
    source_files.update([ROOT/"configs/flux4b_48_one_id.yaml",ROOT/"ba_dit/adapters.py",ROOT/"ba_dit/training.py",ROOT/"ba_dit/checkpoint.py",ROOT/"ba_dit/data/geometry.py",ROOT/"patches/flux2_reference_branch_and_offload.patch",MANIFEST,OLD/"src/configs/CL39_cosmic_null_key_confidence_router_24k.yaml",Path(__file__),ROOT/"runs/review_one_id/status_20261001_0840.json"])
    source_files.update(ROOT/f"runs/review_one_id/validation_change_000000_to_{s:06d}.json" for s in [500,1000,1500])
    manifest=json.loads(MANIFEST.read_text())
    audit={"report_date":"2026-10-01","evidence_cutoff":"2026-10-01T08:41:06Z",
           "project_head":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
           "checkpoint_manifest":manifest,"trainable_parameter_count":sum(math.prod(x) for x in manifest["parameters"].values()),
           "source_hashes":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_files)},
           "pages":[{"number":i,"title":title} for i,(_,title) in enumerate(pages,1)]}
    (OUT/"source_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    # Add a navigable PDF outline and standalone vector diagrams.
    import pymupdf
    doc=pymupdf.open(target)
    doc.set_toc([[1,title,i] for i,(_,title) in enumerate(pages,1)])
    doc.saveIncr()
    for number,name in [(2,"whole_model"),(4,"branched_attention")]:
        single=pymupdf.open(); single.insert_pdf(doc,from_page=number-1,to_page=number-1)
        single.save(OUT/f"{name}.pdf"); single.close()
    doc.close()
    index='''<!doctype html><meta charset="utf-8"><title>FLUX Base 4B branched attention</title>
<style>body{margin:0;background:#e8edf0;font:16px system-ui;color:#20313b}header{padding:24px 4%;background:#006982;color:white}header a{color:white}main{max-width:1600px;margin:auto;padding:20px}img{width:100%;box-shadow:0 2px 10px #0002;margin:16px 0}nav{columns:2;padding:20px 0;line-height:1.7}a{color:#006982}h2{font-size:18px;margin-top:40px}</style>
<header><h1>FLUX Base 4B — current branched attention architecture</h1><p>1 October 2026 · exact active run · CL39 comparison · source-linked code excerpts</p><p><a href="flux4b_branched_attention_architecture.pdf">Download the 16-page PDF</a> · <a href="source_audit.json">Source and checkpoint audit</a></p></header><main><nav>'''
    index+=''.join(f'<a href="#page{i}">{i}. {html.escape(title)}</a><br>' for i,(_,title) in enumerate(pages,1))+'</nav>'
    index+=''.join(f'<section id="page{i}"><h2>{i}. {html.escape(title)} · <a href="page_{i:02d}.svg">SVG</a> · <a href="page_{i:02d}.png">PNG</a></h2><img loading="lazy" src="page_{i:02d}.svg" alt="{html.escape(title)}"></section>' for i,(_,title) in enumerate(pages,1))+'</main>'
    (OUT/"index.html").write_text(index)
    print(json.dumps({"pdf":str(target),"pages":len(pages),"parameter_count":audit['trainable_parameter_count']}))


if __name__ == "__main__":
    main()
