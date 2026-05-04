"""
Figure 1 — QEC codes and hardware mappings (matplotlib version).

2 rows × 3 columns:
  (a) Color code d=3 stabilizers       (b) Heavy-hex surface code d=3       (c) Rotated surface code d=3
  (d) IonQ Forte-1 (color code)        (e) IBM Heron R3 (heavy-hex patch)   (f) IBM Nighthawk (rotated patch)

Top row: abstract stabilizer view of each code.
Bottom row: same code mapped onto its hardware platform.
Top → bottom dashed arrows indicate the code → hardware mapping.

Output: fig1_codes.{png,pdf}
"""
import sys
sys.path.insert(0, '.')
from fig_common import *
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, Polygon, Rectangle, Wedge,
                                FancyBboxPatch, ConnectionPatch)
from matplotlib.transforms import Affine2D

# ============================================================
# Style — matched to the TikZ palette
# ============================================================
C = {
    'R':      '#D55E00',
    'G':      '#009E73',
    'B':      '#0072B2',
    'Zanc':   '#4477AA',
    'Xanc':   '#CC6677',
    'bridge': '#BBBBBB',
    'data':   '#666666',
    'idle':   '#E8E8E8',
    'idleE':  '#888888',
    'title':  '#333333',
}

PANEL_W = 5.4
PANEL_H = 6.2

# Marker sizes
DATA_R     = 0.11
ANC_R      = 0.10
BRIDGE_R   = 0.10
IDLE_R     = 0.05


# ============================================================
# Generic drawing helpers
# ============================================================
def panel_frame(ax):
    ax.set_xlim(0, PANEL_W)
    ax.set_ylim(0, PANEL_H)
    ax.set_aspect('equal')
    ax.axis('off')


def panel_title(ax, label, title):
    ax.text(PANEL_W / 2, PANEL_H - 0.15,
            f'$\\mathbf{{({label})}}$  {title}',
            ha='center', va='top', fontsize=20, color=C['title'])


def panel_subtitle(ax, text):
    ax.text(PANEL_W / 2, PANEL_H - 1.05, text,
            ha='center', va='top', fontsize=14,
            color=C['title'], style='italic')


def panel_caption(ax, text, y=0.50):
    """Bottom-center small italic caption, placed inside panel area."""
    ax.text(PANEL_W / 2, y, text,
            ha='center', va='bottom', fontsize=14,
            color=C['title'], style='italic')


def add_data(ax, x, y, scale=1.0):
    ax.add_patch(Circle((x, y), DATA_R * scale,
                        facecolor=C['data'],
                        edgecolor='black', linewidth=0.5, zorder=5))


def add_anc(ax, x, y, kind, boundary=False, scale=1.0):
    color = C['Zanc' if kind == 'Z' else 'Xanc']
    if boundary:
        ax.add_patch(Circle((x, y), ANC_R * scale,
                            facecolor='white',
                            edgecolor=color, linewidth=1.1, zorder=5))
    else:
        ax.add_patch(Circle((x, y), ANC_R * scale,
                            facecolor=color, alpha=0.55,
                            edgecolor='black', linewidth=0.4, zorder=5))


def add_bridge(ax, x, y, scale=1.0):
    s = BRIDGE_R * scale
    diamond = Polygon(
        [(x, y + s), (x + s, y), (x, y - s), (x - s, y)],
        closed=True, facecolor=C['bridge'], alpha=0.55,
        edgecolor='black', linewidth=0.4, zorder=5)
    ax.add_patch(diamond)


def add_idle(ax, x, y):
    ax.add_patch(Circle((x, y), IDLE_R, facecolor=C['idle'],
                        edgecolor=C['idleE'], linewidth=0.3,
                        alpha=0.6, zorder=2))


def add_edge(ax, p1, p2, alpha=0.45, lw=0.4):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
            color='black', alpha=alpha, linewidth=lw, zorder=2)


# ============================================================
# Panel (a) — Color code d=3
# ============================================================
def draw_panel_a(ax):
    panel_frame(ax)
    panel_title(ax, 'a', 'Color code, $d=3$')
    # legend (colored bullets) — centered horizontally
    legend_items = [(C['R'], 'R'), (C['G'], 'G'), (C['B'], 'B')]
    y_leg = PANEL_H - 1.05
    x = 1.65
    for col, lbl in legend_items:
        ax.plot(x, y_leg, 'o', color=col, markersize=5,
                markeredgecolor='black', markeredgewidth=0.3)
        ax.text(x + 0.15, y_leg, lbl, va='center',
                fontsize=16, color=C['title'])
        x += 0.55
    ax.text(x - 0.05, y_leg, 'faces',
            va='center', fontsize=16, color=C['title'])

    # 7 data qubit positions (Steane octagonal layout — balanced aspect)
    q0 = (2.7, 4.40)   # top apex
    q1 = (1.10, 3.50)  # upper-left
    q3 = (4.30, 3.50)  # upper-right
    q2 = (2.7, 2.65)   # center (shared by all 3 faces)
    q4 = (1.10, 1.80)  # lower-left
    q6 = (4.30, 1.80)  # lower-right
    q5 = (2.7, 0.95)   # bottom apex
    coords = [q0, q1, q2, q3, q4, q5, q6]

    # Weight-4 plaquettes (R/G/B quadrilaterals)
    ax.add_patch(Polygon([q0, q1, q2, q3], closed=True,
                         facecolor=C['R'], alpha=0.22,
                         edgecolor=C['R'], linewidth=0.8))
    ax.add_patch(Polygon([q1, q4, q5, q2], closed=True,
                         facecolor=C['G'], alpha=0.22,
                         edgecolor=C['G'], linewidth=0.8))
    ax.add_patch(Polygon([q3, q2, q5, q6], closed=True,
                         facecolor=C['B'], alpha=0.22,
                         edgecolor=C['B'], linewidth=0.8))

    for x_, y_ in coords:
        add_data(ax, x_, y_)

    # # Anchor labels
    # ax.text(q0[0], q0[1] + 0.22, '$q_0$', ha='center', va='bottom',
    #         fontsize=12, color=C['title'])
    # ax.text(q3[0] + 0.22, q3[1], '$q_3$', ha='left', va='center',
    #         fontsize=12, color=C['title'])
    # ax.text(q4[0] - 0.22, q4[1], '$q_4$', ha='right', va='center',
    #         fontsize=12, color=C['title'])

    panel_caption(ax, '7 data + 3 ancillas on faces (R,G,B)',
                  y=0.20)


# ============================================================
# Panel (b) — Heavy-hex surface code d=3 (correct topology)
# ============================================================
def draw_panel_b(ax):
    panel_frame(ax)
    panel_title(ax, 'b', 'Heavy-hex surface code, $d=3$')
    # legend — centered horizontally
    y_leg = PANEL_H - 1.05
    ax.plot(1.05, y_leg, 'o', color=C['Zanc'], markersize=5,
            markeredgecolor='black', markeredgewidth=0.3)
    ax.text(1.20, y_leg, 'Z-stab', va='center', fontsize=16,
            color=C['title'])
    ax.plot(2.30, y_leg, 'o', color=C['Xanc'], markersize=5,
            markeredgecolor='black', markeredgewidth=0.3)
    ax.text(2.45, y_leg, 'X-stab', va='center', fontsize=16,
            color=C['title'])
    ax.plot(3.55, y_leg, 'D', color=C['bridge'], markersize=5,
            markeredgecolor='black', markeredgewidth=0.3)
    ax.text(3.70, y_leg, 'bridge', va='center', fontsize=16,
            color=C['title'])

    # Data qubits 3x3 grid (shifted +0.20 in x to center within panel)
    # Lattice flipped vertically: row 0 at bottom, row 2 at top
    rows_y = {0: 1.60, 1: 2.85, 2: 4.10}
    cols_x = {0: 1.15, 1: 2.70, 2: 4.25}
    hd = {(r, c): (cols_x[c], rows_y[r]) for r in rows_y for c in cols_x}

    # Bridges (horizontal between adjacent data per row)
    bridges_L = {r: (1.925, rows_y[r]) for r in rows_y}   # cols 0-1
    bridges_R = {r: (3.475, rows_y[r]) for r in rows_y}   # cols 1-2

    # Ancillas — heavy-hex constraint (alternating cols per row gap)
    hzL = (1.15, 2.225)   # row 0-1 gap, col 0  (now lower half)
    hzR = (4.25, 2.225)   # row 0-1 gap, col 2  (now lower half)
    hxC = (2.70, 3.475)   # row 1-2 gap, col 1  (now upper half)

    # Plaquette tints (drawn first) — Z bands now span the lower half
    ax.add_patch(Rectangle((0.85, 1.30), 2.85 - 0.65, 3.15 - 1.30,
                           facecolor=C['Zanc'], alpha=0.18, zorder=1))
    ax.add_patch(Rectangle((2.35, 1.30), 4.35 - 2.15, 3.15 - 1.30,
                           facecolor=C['Zanc'], alpha=0.18, zorder=1))
    ax.add_patch(Rectangle((0.85, 2.55), 2.85 - 0.65, 4.40 - 2.55,
                           facecolor=C['Xanc'], alpha=0.20, zorder=1))
    ax.add_patch(Rectangle((2.35, 2.55), 4.35 - 2.15, 4.40 - 2.55,
                           facecolor=C['Xanc'], alpha=0.20, zorder=1))

    # Heavy-hex edges
    # Horizontal: data-bridge-data-bridge-data per row
    for r in rows_y:
        add_edge(ax, hd[(r, 0)], bridges_L[r])
        add_edge(ax, bridges_L[r], hd[(r, 1)])
        add_edge(ax, hd[(r, 1)], bridges_R[r])
        add_edge(ax, bridges_R[r], hd[(r, 2)])
    # Vertical row 0-1: only cols 0 and 2 (via Z-ancillas)
    add_edge(ax, hd[(0, 0)], hzL); add_edge(ax, hzL, hd[(1, 0)])
    add_edge(ax, hd[(0, 2)], hzR); add_edge(ax, hzR, hd[(1, 2)])
    # Vertical row 1-2: only col 1 (via X-ancilla)
    add_edge(ax, hd[(1, 1)], hxC); add_edge(ax, hxC, hd[(2, 1)])

    # Nodes
    for pos in hd.values():
        add_data(ax, *pos)
    for r in rows_y:
        add_bridge(ax, *bridges_L[r])
        add_bridge(ax, *bridges_R[r])
    add_anc(ax, *hzL, kind='Z')
    add_anc(ax, *hzR, kind='Z')
    add_anc(ax, *hxC, kind='X')

    # Plaquette labels (Z labels now in the lower band, X labels in the upper band)
    ax.text(1.95, 2.20, '$Z_1$', ha='center', va='center',
            fontsize=16, color=C['Zanc'])
    ax.text(3.45, 2.20, '$Z_2$', ha='center', va='center',
            fontsize=16, color=C['Zanc'])
    ax.text(1.95, 3.50, '$X_1$', ha='center', va='center',
            fontsize=16, color=C['Xanc'])
    ax.text(3.45, 3.50, '$X_2$', ha='center', va='center',
            fontsize=16, color=C['Xanc'])

    panel_caption(ax, '9 data + 6 bridges + 3 ancillas', y=0.20)


# ============================================================
# Panel (c) — Rotated surface code d=3
# ============================================================
def draw_panel_c(ax):
    panel_frame(ax)
    panel_title(ax, 'c', 'Rotated surface code, $d=3$')
    # legend — centered horizontally
    y_leg = PANEL_H - 1.05
    ax.plot(0.35, y_leg, 'o', color=C['Zanc'], markersize=5,
            markeredgecolor='black', markeredgewidth=0.3)
    ax.text(0.50, y_leg, 'Z-stab', va='center', fontsize=16,
            color=C['title'])
    ax.plot(1.60, y_leg, 'o', color=C['Xanc'], markersize=5,
            markeredgecolor='black', markeredgewidth=0.3)
    ax.text(1.75, y_leg, 'X-stab', va='center', fontsize=16,
            color=C['title'])
    ax.plot(2.85, y_leg, 'o', mfc='white',
            markeredgecolor=C['title'], markeredgewidth=0.8, markersize=5)
    ax.text(3.00, y_leg, 'boundary (weight-2)',
            va='center', fontsize=16, color=C['title'])

    # 9 data on 3x3 grid (q0 at top-left, row-major)
    rows_y = {0: 4.15, 1: 3.00, 2: 1.85}   # row 0 = top, row 2 = bottom
    cols_x = {0: 1.30, 1: 2.70, 2: 4.10}
    sd = {(r, c): (cols_x[c], rows_y[r]) for r in rows_y for c in cols_x}

    # Bulk ancillas (centers of the 4 bulk plaquettes)
    #   X-bulk top-LEFT cell : [q0,q1,q4,q3]
    #   Z-bulk top-RIGHT cell: [q1,q2,q5,q4]
    #   Z-bulk bot-LEFT cell : [q3,q4,q7,q6]
    #   X-bulk bot-RIGHT cell: [q4,q5,q8,q7]
    sx1 = (2.00, 3.575)   # X-bulk top-LEFT
    sz1 = (3.40, 3.575)   # Z-bulk top-RIGHT
    sz2 = (2.00, 2.425)   # Z-bulk bot-LEFT
    sx2 = (3.40, 2.425)   # X-bulk bot-RIGHT

    # Boundary ancillas (just outside grid)
    #   X-bnd LEFT  (q3,q6) : left edge bottom half
    #   X-bnd RIGHT (q2,q5) : right edge top half
    #   Z-bnd TOP   (q0,q1) : top edge left half
    #   Z-bnd BOT   (q7,q8) : bottom edge right half
    sx_L = (cols_x[0] - 0.65, (rows_y[1] + rows_y[2]) / 2)   # (0.65, 2.425)
    sx_R = (cols_x[2] + 0.65, (rows_y[0] + rows_y[1]) / 2)   # (4.75, 3.575)
    sz_T = ((cols_x[0] + cols_x[1]) / 2, rows_y[0] + 0.65)   # (2.00, 4.80)
    sz_B = ((cols_x[1] + cols_x[2]) / 2, rows_y[2] - 0.65)   # (3.40, 1.20)

    # Bulk plaquette tints (squares around the 4 cells)
    ax.add_patch(Polygon([sd[(0, 0)], sd[(0, 1)], sd[(1, 1)], sd[(1, 0)]],
                         facecolor=C['Xanc'], alpha=0.20, zorder=1))   # top-LEFT (X)
    ax.add_patch(Polygon([sd[(0, 1)], sd[(0, 2)], sd[(1, 2)], sd[(1, 1)]],
                         facecolor=C['Zanc'], alpha=0.20, zorder=1))   # top-RIGHT (Z)
    ax.add_patch(Polygon([sd[(1, 0)], sd[(1, 1)], sd[(2, 1)], sd[(2, 0)]],
                         facecolor=C['Zanc'], alpha=0.20, zorder=1))   # bot-LEFT (Z)
    ax.add_patch(Polygon([sd[(1, 1)], sd[(1, 2)], sd[(2, 2)], sd[(2, 1)]],
                         facecolor=C['Xanc'], alpha=0.20, zorder=1))   # bot-RIGHT (X)

    # Boundary D-shape semicircles (Wedge), centered on the edge midpoint,
    # bulging outward away from the grid.
    # X-bnd LEFT: between sd10 and sd20, points LEFT
    cx, cy = (sd[(1, 0)][0] + sd[(2, 0)][0]) / 2, (sd[(1, 0)][1] + sd[(2, 0)][1]) / 2
    ax.add_patch(Wedge((cx, cy), 0.575, 90, 270,
                       facecolor=C['Xanc'], alpha=0.20,
                       edgecolor='#7a3838', linewidth=1.2, zorder=2))
    # X-bnd RIGHT: between sd02 and sd12, points RIGHT
    cx, cy = (sd[(0, 2)][0] + sd[(1, 2)][0]) / 2, (sd[(0, 2)][1] + sd[(1, 2)][1]) / 2
    ax.add_patch(Wedge((cx, cy), 0.575, -90, 90,
                       facecolor=C['Xanc'], alpha=0.20,
                       edgecolor='#7a3838', linewidth=1.2, zorder=2))
    # Z-bnd TOP: between sd00 and sd01, points UP
    cx, cy = (sd[(0, 0)][0] + sd[(0, 1)][0]) / 2, (sd[(0, 0)][1] + sd[(0, 1)][1]) / 2
    ax.add_patch(Wedge((cx, cy), 0.700, 0, 180,
                       facecolor=C['Zanc'], alpha=0.20,
                       edgecolor='#1f3a55', linewidth=1.2, zorder=2))
    # Z-bnd BOT: between sd21 and sd22, points DOWN
    cx, cy = (sd[(2, 1)][0] + sd[(2, 2)][0]) / 2, (sd[(2, 1)][1] + sd[(2, 2)][1]) / 2
    ax.add_patch(Wedge((cx, cy), 0.700, 180, 360,
                       facecolor=C['Zanc'], alpha=0.20,
                       edgecolor='#1f3a55', linewidth=1.2, zorder=2))

    # Square-grid lattice edges
    for r in rows_y:
        add_edge(ax, sd[(r, 0)], sd[(r, 1)])
        add_edge(ax, sd[(r, 1)], sd[(r, 2)])
    for c in cols_x:
        add_edge(ax, sd[(0, c)], sd[(1, c)])
        add_edge(ax, sd[(1, c)], sd[(2, c)])

    # Nodes
    for pos in sd.values():
        add_data(ax, *pos)
    add_anc(ax, *sx1, kind='X'); add_anc(ax, *sx2, kind='X')
    add_anc(ax, *sz1, kind='Z'); add_anc(ax, *sz2, kind='Z')
    # Boundary ancillas (open ring)
    add_anc(ax, *sx_L, kind='X', boundary=True)
    add_anc(ax, *sx_R, kind='X', boundary=True)
    add_anc(ax, *sz_T, kind='Z', boundary=True)
    add_anc(ax, *sz_B, kind='Z', boundary=True)

    panel_caption(ax, '9 data + 4 bulk ancillas + 4 boundary ancillas', y=0.20)


# ============================================================
# Panel (d) — IonQ Forte-1 (color code with all-to-all)
# ============================================================
def draw_panel_d(ax):
    panel_frame(ax)
    panel_title(ax, 'd', 'IonQ Forte-1 (color code, $d=3$)')
    panel_subtitle(ax, 'trapped-ion, all-to-all (logical view)')

    # Steane octagonal layout (matches panel a — balanced aspect)
    q0 = (2.7, 4.40)
    q1 = (1.10, 3.50)
    q3 = (4.30, 3.50)
    q2 = (2.7, 2.65)
    q4 = (1.10, 1.80)
    q6 = (4.30, 1.80)
    q5 = (2.7, 0.95)
    coords = [q0, q1, q2, q3, q4, q5, q6]   # indexed by qubit number

    # Weight-4 plaquettes
    ax.add_patch(Polygon([q0, q1, q2, q3], closed=True,
                         facecolor=C['R'], alpha=0.20,
                         edgecolor=C['R'], linewidth=0.7))
    ax.add_patch(Polygon([q1, q4, q5, q2], closed=True,
                         facecolor=C['G'], alpha=0.20,
                         edgecolor=C['G'], linewidth=0.7))
    ax.add_patch(Polygon([q3, q2, q5, q6], closed=True,
                         facecolor=C['B'], alpha=0.20,
                         edgecolor=C['B'], linewidth=0.7))

    # All-to-all dashed connections (cross-plaquette pairs)
    pairs = [(0, 4), (0, 5), (0, 6), (1, 6), (3, 4), (4, 6)]
    for i, j in pairs:
        ax.plot([coords[i][0], coords[j][0]],
                [coords[i][1], coords[j][1]],
                color='black', alpha=0.35, linewidth=0.5,
                linestyle='--', zorder=1)

    # Data qubits + face-center ancillas (centroid of each 4-qubit face)
    for x, y in coords:
        add_data(ax, x, y)
    iaR = ((q0[0] + q1[0] + q2[0] + q3[0]) / 4,
           (q0[1] + q1[1] + q2[1] + q3[1]) / 4)
    iaG = ((q1[0] + q4[0] + q5[0] + q2[0]) / 4,
           (q1[1] + q4[1] + q5[1] + q2[1]) / 4)
    iaB = ((q3[0] + q2[0] + q5[0] + q6[0]) / 4,
           (q3[1] + q2[1] + q5[1] + q6[1]) / 4)
    ax.add_patch(Circle(iaR, ANC_R, facecolor=C['R'], alpha=0.55,
                        edgecolor='black', linewidth=0.4, zorder=5))
    ax.add_patch(Circle(iaG, ANC_R, facecolor=C['G'], alpha=0.55,
                        edgecolor='black', linewidth=0.4, zorder=5))
    ax.add_patch(Circle(iaB, ANC_R, facecolor=C['B'], alpha=0.55,
                        edgecolor='black', linewidth=0.4, zorder=5))

    panel_caption(ax, '7 data + 3 ancillas', y=0.20)


# ============================================================
# Panel (e) — IBM Heron R3 heavy-hex lattice with patch
# ============================================================
def draw_panel_e(ax):
    panel_frame(ax)
    panel_title(ax, 'e', 'IBM Heron R3, $d=3$')
    panel_subtitle(ax, 'heavy-hex lattice')

    # Lattice geometry: 5 vertex rows × 5 vertex cols
    dx, dy = 0.92, 0.92
    xoff, yoff = 0.86, 0.95

    # Vertex positions (all 25 sites)
    V = {(r, c): (xoff + c * dx, yoff + r * dy)
         for r in range(5) for c in range(5)}
    # Horizontal link positions (between adjacent vertices in same row)
    LH = {(r, c): (xoff + (c + 0.5) * dx, yoff + r * dy)
          for r in range(5) for c in range(4)}
    # Vertical link positions — heavy-hex parity rule
    # Row gap r→r+1: cols {0,2,4} if r even; cols {1,3} if r odd
    LV = {}
    for r in range(4):
        cols = [0, 2, 4] if r % 2 == 0 else [1, 3]
        for c in cols:
            LV[(r, c)] = (xoff + c * dx, yoff + (r + 0.5) * dy)

    # Background lattice edges + idle markers
    for pos in V.values():
        add_idle(ax, *pos)
    for (r, c), pos in LH.items():
        add_idle(ax, *pos)
        add_edge(ax, V[(r, c)], pos, alpha=0.35)
        add_edge(ax, pos, V[(r, c + 1)], alpha=0.35)
    for (r, c), pos in LV.items():
        add_idle(ax, *pos)
        add_edge(ax, V[(r, c)], pos, alpha=0.35)
        add_edge(ax, pos, V[(r + 1, c)], alpha=0.35)

    # Patch: rows 1..3 cols 1..3 (centered in lattice)
    # Data qubits = vertices in patch
    for r in (1, 2, 3):
        for c in (1, 2, 3):
            add_data(ax, *V[(r, c)])
    # Bridges = horizontal links inside patch
    for r in (1, 2, 3):
        add_bridge(ax, *LH[(r, 1)])
        add_bridge(ax, *LH[(r, 2)])
    # Z-ancillas at row gap 1-2, cols 1, 3 (heavy-hex parity allows)
    add_anc(ax, *LV[(1, 1)], kind='Z')
    add_anc(ax, *LV[(1, 3)], kind='Z')
    # X-ancilla at row gap 2-3, col 2
    add_anc(ax, *LV[(2, 2)], kind='X')

    # Patch boundary (dashed rectangle)
    x0 = V[(1, 1)][0] - 0.30
    y0 = V[(1, 1)][1] - 0.55
    x1 = V[(3, 3)][0] + 0.30
    y1 = V[(3, 3)][1] + 0.30
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                           facecolor='none', edgecolor='black',
                           linewidth=0.7, linestyle='--', zorder=3))

    panel_caption(ax, '9 data + 6 bridges + 3 ancillas', y=0.20)


# ============================================================
# Panel (f) — IBM Nighthawk square lattice with 45° patch
# ============================================================
def draw_panel_f(ax):
    panel_frame(ax)
    panel_title(ax, 'f', 'IBM Nighthawk, $d=3$')
    panel_subtitle(ax, 'square lattice, $45^{\\circ}$ rotation')

    # Background lattice 5 rows × 7 cols (square 4-neighbor) — outer ring trimmed
    sx, sy = 0.55, 0.55
    sxoff, syoff = 0.50, 1.10
    R_RANGE = range(1, 6)
    C_RANGE = range(1, 8)

    G = {(r, c): (sxoff + c * sx, syoff + r * sy)
         for r in R_RANGE for c in C_RANGE}

    for pos in G.values():
        add_idle(ax, *pos)
    for r in R_RANGE:
        for c in range(1, 7):
            add_edge(ax, G[(r, c)], G[(r, c + 1)], alpha=0.35)
    for r in range(1, 5):
        for c in C_RANGE:
            add_edge(ax, G[(r, c)], G[(r + 1, c)], alpha=0.35)

    # Patch (diamond, 45° rotated d=3 surface code)
    # Data qubits on diamond vertices (lattice coords; user-spec coords + (1,2))
    data_cells = [(1, 4), (2, 3), (2, 5), (3, 2), (3, 4),
                  (3, 6), (4, 3), (4, 5), (5, 4)]
    # Bulk ancillas — match select_best_patch(d=3) output
    #   X-bulk: top-center (1,2), bot-center (3,2)  → lattice (2,4), (4,4)
    #   Z-bulk: left-center (2,1), right-center (2,3) → lattice (3,3), (3,5)
    bulk_X = [(2, 4), (4, 4)]
    bulk_Z = [(3, 3), (3, 5)]
    # Boundary ancillas (open rings) — asymmetric placement from hardware patch
    #   Z-bnd: top-left (0,1), bot-right (4,3)  → lattice (1,3), (5,5)
    #   X-bnd: top-right (1,4), bot-left (3,0)  → lattice (2,6), (4,2)
    bnd_Z = [(1, 3), (5, 5)]
    bnd_X = [(2, 6), (4, 2)]

    # Bulk plaquette tints (square between 4 data qubits forming each cell)
    cells = [
        ((1, 4), (2, 3), (3, 4), (2, 5), C['Xanc']),   # top X-bulk  (q11,q2,q22,q13)
        ((2, 3), (3, 2), (4, 3), (3, 4), C['Zanc']),   # left Z-bulk (q11,q20,q31,q22)
        ((2, 5), (3, 4), (4, 5), (3, 6), C['Zanc']),   # right Z-bulk(q13,q22,q33,q24)
        ((3, 4), (4, 3), (5, 4), (4, 5), C['Xanc']),   # bot X-bulk  (q22,q31,q42,q33)
    ]
    for p1, p2, p3, p4, color in cells:
        ax.add_patch(Polygon([G[p1], G[p2], G[p3], G[p4]],
                             facecolor=color, alpha=0.20, zorder=1))

    # Boundary D-shapes — one semicircle per occupied outer edge.
    # Each pair (p1, p2) is two adjacent diamond-vertex data qubits;
    # wedge bulges outward away from the diamond center G[(3,4)].
    bnd_edges = [
        ((1, 4), (2, 3), 'Z'),   # top-left  outer edge — Z (paired with bnd_Z (1,3))
        ((2, 5), (3, 6), 'X'),   # top-right outer edge — X (paired with bnd_X (2,6))
        ((3, 2), (4, 3), 'X'),   # bot-left  outer edge — X (paired with bnd_X (4,2))
        ((5, 4), (4, 5), 'Z'),   # bot-right outer edge — Z (paired with bnd_Z (5,5))
    ]
    for p1, p2, kind in bnd_edges:
        a = np.array(G[p1])
        b = np.array(G[p2])
        mid = (a + b) / 2
        # Direction from midpoint pointing outward (perpendicular to edge, away from diamond center)
        diamond_center = np.array(G[(3, 4)])
        outward = mid - diamond_center
        outward /= np.linalg.norm(outward)
        # The edge direction
        edge_vec = b - a
        edge_len = np.linalg.norm(edge_vec)
        # Wedge centered on midpoint, radius spans the bulge
        # angle of wedge: edge tangent ± 90° on the outward side
        # tangent angle from a to b
        tangent_angle = np.degrees(np.arctan2(edge_vec[1], edge_vec[0]))
        # The wedge runs from angle theta1 to theta2, on the outward half
        # outward angle:
        outward_angle = np.degrees(np.arctan2(outward[1], outward[0]))
        # semicircle on outward side: from (outward - 90°) to (outward + 90°)
        theta1 = outward_angle - 90
        theta2 = outward_angle + 90
        color = C['Zanc'] if kind == 'Z' else C['Xanc']
        edge_color = '#1f3a55' if kind == 'Z' else '#7a3838'
        ax.add_patch(Wedge(tuple(mid), edge_len / 2 * 0.95, theta1, theta2,
                           facecolor=color, alpha=0.20,
                           edgecolor=edge_color, linewidth=1.2, zorder=2))

    # Data qubits
    for cell in data_cells:
        add_data(ax, *G[cell])
    # Bulk ancillas
    for cell in bulk_Z:
        add_anc(ax, *G[cell], kind='Z')
    for cell in bulk_X:
        add_anc(ax, *G[cell], kind='X')
    # Boundary ancillas (open rings)
    for cell in bnd_Z:
        add_anc(ax, *G[cell], kind='Z', boundary=True)
    for cell in bnd_X:
        add_anc(ax, *G[cell], kind='X', boundary=True)

    # 45° patch boundary (rotated square)
    cx, cy = G[(3, 4)]
    side = 1.20
    rect = Rectangle((cx - side, cy - side), 2 * side, 2 * side,
                     facecolor='none', edgecolor='black',
                     linewidth=0.7, linestyle='--', zorder=3)
    rect.set_transform(Affine2D().rotate_deg_around(cx, cy, 45) + ax.transData)
    ax.add_patch(rect)

    panel_caption(ax, '9 data + 4 bulk ancillas + 4 boundary ancillas', y=0.20)


# ============================================================
# Compose figure
# ============================================================
def main():
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.9))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,
                        hspace=0.04, wspace=0.05)

    draw_panel_a(axes[0, 0])
    draw_panel_b(axes[0, 1])
    draw_panel_c(axes[0, 2])
    draw_panel_d(axes[1, 0])
    draw_panel_e(axes[1, 1])
    draw_panel_f(axes[1, 2])

    for ext in ('png', 'pdf'):
        out = DATA_DIR / f'fig1_codes.{ext}'
        fig.savefig(out, dpi=220, bbox_inches='tight', pad_inches=0.05)
        print(f'Saved: {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()