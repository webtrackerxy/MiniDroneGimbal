"""Three-axis test gimbal for a small egg-bodied quadcopter.

A portal frame holds a wide, flat octagonal outer ring on a vertical axis (yaw),
that ring holds a long, shallow inner ring on the axis across the portal
(pitch), and the inner ring holds the drone's cradle on the front-to-back axis
(roll).

Rig axes: X across the portal, Y front to back, Z up, origin at the drone's
centre of gravity. Each pivot is the same stack, from the turning part out:

    boss -> 3.8 mm nub -> inner race of an MR63ZZ (pressed into the outer part)
         -> printed shim -> M3 x 8 socket head, through the boss into an M3 nut

The nut pushes into a slot in the boss, open on the side printed uppermost, and
pulls against the boss's end wall. The nub and the shim touch the inner race only: a bare M3 head is 5.5 mm wide
and would clamp the outer race too.

    python mini_drone_gimbal.py check     # sweep all three axes for collisions
    python mini_drone_gimbal.py stl       # stl/*.stl + 3mf/*.3mf
    python mini_drone_gimbal.py html      # preview.html from preview_template.html
    python mini_drone_gimbal.py all
    python mini_drone_gimbal.py v1.1      # check one version, write all its parts, a BOM and a preview to v1.1/
    python mini_drone_gimbal.py versions  # the same for v1.0, v1.1 and v1.2

Needs: pip install numpy manifold3d trimesh networkx lxml

Every part prints without supports, in the orientation it is exported in, on a
256 mm bed. The rings print whole; the top bar and base are cut into sections
that lap side by side and bolt together with M3s.
"""

from __future__ import annotations

import base64
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from manifold3d import CrossSection, JoinType, Manifold, set_circular_segments

set_circular_segments(64)
OUT = Path(__file__).parent
VERSION = "1.2"  # 1.1: slip ring replaces the top pivot bearing; 1.2: bearing on the slip tube


VERSIONS = ("1.0", "1.1", "1.2")


def has_slip_ring():
    """v1.0 has a plain MR63ZZ pivot at the top, the same as the one in the base."""
    return VERSION != "1.0"


def has_slip_bearing():
    """v1.2 adds a 6702ZZ in slip_mount round the tube."""
    return VERSION == "1.2"


def set_version(version):
    """Switch every part to an earlier version's design."""
    global VERSION, SLIP_FOOT_T, SLIP_WINDOW
    VERSION = version
    # v1.1's foot had no bearing pocket, so it was thinner and the window taller. The flange stays put.
    SLIP_FOOT_T, SLIP_WINDOW = (6.5, 7.5) if has_slip_bearing() else (4.0, 10.0)

# Drone, measured with a ruler along the prop tips: 140 mm edge to edge. Body 30 mm tall, measured.
# Centre of gravity is an estimate.
MOTOR_SPACING = 85.0  # square, motor to motor along each side
PROP_D = 56.0
EGG_W, EGG_L, EGG_H = 36.0, 50.0, 30.0  # height measured; long axis on Y
EGG_DIP = 6.0  # how far the egg drops through the tray's seat, worked out from its shape
ARM_Z = (-14.5, -8.5)  # read from a side photo, like the prop and motor heights below
PROP_Z = -3.5
MOTOR_BOTTOM = -38.0
DRONE = MOTOR_SPACING + PROP_D + 1  # square swept by the props
DRONE_H = 50.0
PROP_TOP = 20.0  # top of the prop sweep above the centre of gravity, estimated
CG_ABOVE_TRAY = 20.0  # roll/pitch axes pass this far above the tray's top face
CLEAR = 5.0  # spare round the drone's furthest corner

# MR63ZZ bearing: 3 bore x 6 OD x 2.5 wide.
BEARING_D = 6.1  # printed holes come out small; bearing_fit_test.stl tunes this
BEARING_W = 2.5
PAD_T = 2.3  # wall holding the bearing: it stands 0.1 proud on both faces
NUB_D = 3.8  # on the inner race (about 4 OD), clear of the outer race
NUB_L = 0.4
M3_THREAD_D = 2.6  # frame screws self-thread into the uprights
PIVOT_HOLE_D = 3.3
PIVOT_HOLE_DEPTH = 5.5  # past the nut: an M3 x 8 ends here
NUT_AF, NUT_T = 5.7, 2.6  # M3 nut slot, with clearance
NUT_WALL = 2.0  # boss end wall the nut pulls against
HEAD_D, HEAD_H = 5.5, 3.0  # M3 socket head
SHIM_OD, SHIM_ID, SHIM_T = 3.9, 3.3, 0.6

# Rings.
RING_T = 3.0  # radial
RING_H = 10.0  # along the ring's axis, as printed
RING_GAP = 1.2  # inner ring's headroom inside the outer ring
INNER_MARGIN = 20.0  # drone edge to the inner ring, front and back
INNER_STRETCH = 1.1  # inner ring's width over its depth
BOSS_L = 6.0  # ring to next part: room for the nut
INNER_CHAMFER, OUTER_CHAMFER = 30.0, 30.0
BOSS_W = 8.0
ARM_W = 3.0  # cradle arm web, between the props
ARM_STRIP = 8.0  # height of the arm's bottom strip
SENSOR_Y = 20.0  # sensor mount screws through the tray, on the arms' centreline
SENSOR_LEG = 10.0  # clears the egg where it dips through the seat
SENSOR_W = 28.0
SENSOR_T = 3.0
TRAY_HALF = 26.0
COVER_WALL = 1.2
COVER_CLEAR = 4.0  # egg to cover, sideways, for tape, wires and an egg wider than measured
COVER_FOAM = 2.0  # foam pad between the egg and the crown
COVER_H = 17.0  # rim to crown: the rim stops partway down the arms, so they sit in the notches
COVER_OVERHANG = 35.0  # dome stops getting flatter than this, in degrees, so it prints bare
ARM_NOTCH_W = 8.0  # stops the drone turning in the cover; the arm need not touch
ARM_NOTCH_H = 4.0
CENTRE_HOLE_D = 1.5  # straight through the clip and the cover's crown
CLIP_W = 10.0
CLIP_T = 3.0
CLIP_TOOTH = 1.4
CLIP_RAMP = 1.2  # height of each tooth's lead-in
CLIP_PITCH = 4.5  # leaves 0.3 mm round the 3 mm tray
CLIP_NOTCHES_LONGER = 7  # below the drawn notch
CLIP_NOTCHES_SHORTER = 4  # above it, up the leg as far as the corner
CLIP_TAB = 10.0  # thumb tab past the leg
POST_L = 8.0  # upright at the arm's outer end, along the arm
OUTER_BOSS_W = 16.0  # yaw posts, top and bottom of the outer ring
PAD_W = 16.0

# Bolted sections: everything must fit this bed.
BED = 256.0
BAR_LAP = 24.0
BASE_CUT, TOP_CUT = 44.0, 30.0
M3_BORE_D = 3.3
NUT_RECESS, HEAD_RECESS = 2.6, 3.2

# Frame.
BAR = 24.0  # square section of every frame member
SIDE_GAP = 9.0  # outer ring's sweep to the uprights
SOCKET = 8.0  # upright goes this deep into the base and the top bar
SOCKET_CLEAR = 0.2
M3_CLEAR_D, M3_HEAD_D = 3.4, 6.2
M3_FLOOR = 3.5  # plastic under the head, so an M3 x 12 still reaches the upright
M3_HEAD_H = BAR - SOCKET - M3_FLOOR
RIB_T, RIB_L = 12.0, 30.0  # 45 degree ribs holding the uprights square
CBORE_D = 19.0  # head and key access to the frame's bearings
FRAME_PAD_D = 20.0

# Slip ring at the top pivot: sizes from the SNM022A-12 drawing, check against the real part.
SLIP_FLANGE_D, SLIP_FLANGE_T = 44.5, 2.4
SLIP_PCD = 35.0  # three holes, screws self-thread into the mount
SLIP_BODY_D, SLIP_L = 22.0, 26.0
SLIP_SHOULDER_L = 5.0
SLIP_STEM_D, SLIP_STEM_L = 7.8, 9.0
SLIP_STEM_FIT = 0.3
SLIP_FOOT_T = 6.5  # holds the tube's bearing
SLIP_HEAD_SEAT = 4.0  # under the foot screws' heads
SLIP_WINDOW = 7.5  # room to reach the set screws
SLIP_PLATE_T = 7.0
SLIP_PLATE_D = 50.0
SLIP_FOOT_X = 24.0  # screws into the bar, inside the lap joint at x = 30
SLIP_FOOT_PILOT = 13.0  # an M3 x 16 reaches 12 mm into the bar
SLIP_TUBE_D = 15.0
SLIP_BORE_D = 6.5  # passes the wires and an M3 head
SLIP_SPIGOT_D, SLIP_SPIGOT_L = 18.8, 5.0
SLIP_SPIGOT_BORE = 16.6
SLIP_BEARING_OD, SLIP_BEARING_W = 21.0, 4.0  # 6702ZZ: 15 bore, on the tube
SLIP_BEARING_D = 21.1  # printed holes come out small
SLIP_SHELF_BORE = 17.6  # catches the outer race, clear of the inner race
SLIP_TUBE_GAP = 1.5  # tube top to the slip ring's shoulder
SLIP_FLOOR = 3.0
SLIP_SPIGOT_TUBE_D = 7.0  # tube's lower end, down the centre of the outer ring's top boss
SLIP_SEAT_D = 7.3  # printed holes come out small
SLIP_WIRE_BORE = 4.2  # all the wires pass down the middle of the spigot
SLIP_SPIGOT_END = 101.0  # stops above the wire relief in the ring's inner face
SLIP_RELIEF_Z = 100.5  # relief lets the wires turn clear of the inner ring's screw heads
SLIP_RELIEF_W, SLIP_RELIEF_X = 4.4, 8.0
SLIP_LOCK_Z = 104.75  # side screw's axis: its head clears the ring's top face
SLIP_LOCK_DIMPLE = 0.75  # how far the screw tip sits in the spigot

_cg = CG_ABOVE_TRAY
_drone_r = math.hypot(DRONE / 2, max(_cg, DRONE_H - _cg))  # sweep round the roll axis
INNER_Y = DRONE / 2 + INNER_MARGIN  # inner ring's inner face, front and back
INNER_X = max(math.ceil(INNER_Y * INNER_STRETCH), math.ceil(_drone_r + CLEAR))
_head_top = PAD_T + 0.1 + SHIM_T + HEAD_H  # above a seat's inner face
_prop_r = math.hypot(math.hypot(MOTOR_SPACING, MOTOR_SPACING) / 2 + (DRONE - MOTOR_SPACING) / 2, PROP_TOP)
OUTER_Z = math.ceil(max(INNER_Y + _head_top + RING_GAP, _prop_r + 1.5))
OUTER_X = INNER_X + RING_T + BOSS_L
_outer_head = OUTER_X + _head_top
FRAME_Z = OUTER_Z + RING_T + BOSS_L  # bar's inner face
FRAME_X = math.ceil(_outer_head + SIDE_GAP)  # upright's inner face
UPRIGHT_X = FRAME_X + BAR / 2
FRAME_W = 2 * (FRAME_X + BAR)
UPRIGHT_L = 2 * (FRAME_Z + SOCKET)


def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def cyl_z(d, z0, z1, x=0.0, y=0.0):
    return Manifold.cylinder(z1 - z0, d / 2).translate((x, y, z0))


def cyl_y(d, y0, y1):
    return cyl_z(d, 0, y1 - y0).rotate((-90, 0, 0)).translate((0, y0, 0))


def teardrop_y(d, y0, y1):
    """Round hole along Y that prints without support: pointed towards +Z."""
    r = d / 2
    s = r * math.sin(math.radians(45))
    tip = CrossSection([[(-s, s), (-0.4, r + 0.7), (0.4, r + 0.7), (s, s)]])
    return (
        (CrossSection.circle(r) + tip)
        .extrude(y1 - y0)
        .rotate((90, 0, 0))
        .translate((0, y1, 0))
    )


def nut_slot_y(end_wall_y):
    """Nut seat just behind y = end_wall_y on the axis, open towards +Z."""
    a, c = NUT_AF / 2, NUT_AF / math.sqrt(3)
    outline = [(-a, RING_H), (-a, -c / 2), (0, -c), (a, -c / 2), (a, RING_H)]
    return (
        CrossSection([outline])
        .extrude(NUT_T)
        .rotate((90, 0, 0))
        .translate((0, end_wall_y, 0))
    )


def octagon(hx, hy, c):
    return CrossSection(
        [[
            (hx, -hy + c), (hx, hy - c), (hx - c, hy), (-hx + c, hy),
            (-hx, hy - c), (-hx, -hy + c), (-hx + c, -hy), (hx - c, -hy),
        ]]
    )


def both_ends(part):
    return part + part.rotate((0, 0, 180))


def ring(hx, hy, chamfer, carrier_face, boss_w=BOSS_W):
    """Bearings on local +-Y, bosses on local +-X out to carrier_face, axis Z."""
    h = RING_H / 2
    core = octagon(hx, hy, chamfer)
    body = (core.offset(RING_T, JoinType.Round) - core).extrude(RING_H).translate((0, 0, -h))

    pad = box(-PAD_W / 2, PAD_W / 2, hy - 0.5, hy + max(PAD_T, RING_T), -h, h)
    bearing = teardrop_y(BEARING_D, hy - 1, hy + max(PAD_T, RING_T) + 1)
    clear_inside = box(-PAD_W, PAD_W, hy - 3, hy, -h - 1, h + 1)

    shoulder = carrier_face - NUB_L - 0.1
    boss = box(hx + RING_T - 0.5, shoulder, -boss_w / 2, boss_w / 2, -h, h)
    nub = cyl_y(NUB_D, -0.1, NUB_L).rotate((0, 0, -90)).translate((shoulder, 0, 0))
    screw = cyl_y(PIVOT_HOLE_D, -PIVOT_HOLE_DEPTH, NUB_L + 1) + nut_slot_y(-NUT_WALL)
    screw = screw.rotate((0, 0, -90)).translate((shoulder, 0, 0))

    body = body + both_ends(pad) - both_ends(clear_inside) - both_ends(bearing)
    return body + both_ends(boss + nub) - both_ends(screw)


def hex_z(z0, z1):
    r = NUT_AF / math.sqrt(3)
    pts = [(r * math.cos(math.radians(60 * i)), r * math.sin(math.radians(60 * i))) for i in range(6)]
    return CrossSection([pts]).extrude(z1 - z0).translate((0, 0, z0))


def inner_ring():
    return ring(INNER_X, INNER_Y, INNER_CHAMFER, OUTER_X)


def outer_ring():
    """As printed: bearings on +-X (for the inner ring), bosses on +-Y."""
    part = ring(OUTER_Z, OUTER_X, OUTER_CHAMFER, FRAME_Z, OUTER_BOSS_W).rotate((0, 0, 90))
    if not has_slip_ring():
        return part
    # +Y here is the top of the rig: no nut or nub, the slip tube runs down the centre instead.
    w = OUTER_BOSS_W / 2
    part += box(-w, w, OUTER_Z + RING_T - 0.5, FRAME_Z - NUB_L - 0.1, -RING_H / 2, RING_H / 2)
    part -= teardrop_y(SLIP_SEAT_D, OUTER_Z - 1, FRAME_Z + 1)
    r = SLIP_RELIEF_W / 2
    part -= box(-SLIP_RELIEF_X, SLIP_RELIEF_X, OUTER_Z - 1, SLIP_RELIEF_Z, -r, r)
    lock = cyl_y(M3_THREAD_D, 0, OUTER_BOSS_W).rotate((0, 0, -90)).translate((0, SLIP_LOCK_Z, 0))
    return part - lock


def lap_split(part, cut, rails, z_mid):
    """Cut bars running along X at x = cut, with a side-by-side lap and M3 x 20s.

    rails holds each bar's centre y. Laps stand on edge so neither section has
    an overhang: nut pockets on the +Y side, head recesses on the -Y side.
    """
    half = BAR / 2
    a = box(-999, cut, -999, 999, -999, 999)
    b = box(cut + BAR_LAP, 999, -999, 999, -999, 999)
    for yc in rails:
        a += box(cut, cut + BAR_LAP, yc - half, yc, -999, 999)
        b += box(cut, cut + BAR_LAP, yc, yc + half, -999, 999)
        for x in (cut + 6, cut + 18):
            bolt = teardrop_y(M3_BORE_D, yc - half - 1, yc + half + 1)
            head = teardrop_y(M3_HEAD_D, yc - half - 1, yc - half + HEAD_RECESS)
            nut = hex_z(0, NUT_RECESS + 1).rotate((0, 0, 30)).rotate((90, 0, 0))
            nut = nut.translate((0, yc + half + 1, 0))
            part -= (bolt + head + nut).translate((x, 0, z_mid))
    return part ^ a, part ^ b


def cradle():
    """Tray under the drone; the arms reach the roll axis between the props."""
    tray_top = -CG_ABOVE_TRAY
    tray_bot = tray_top - 3
    half = TRAY_HALF
    shoulder = INNER_Y - NUB_L - 0.1

    tray = box(-half, half, -half, half, tray_bot, tray_top)
    for x in (-20, 20):  # strap goes through these and over the egg
        tray -= box(x - 2, x + 2, -14, 14, tray_bot - 1, tray_top + 1)
    tray -= cyl_z(EGG_W * 0.8, tray_bot - 1, tray_top + 1)  # the egg sits in this
    for y in (-SENSOR_Y, SENSOR_Y):
        tray -= cyl_z(M3_CLEAR_D, tray_bot - 1, tray_top + 1, y=y)

    # Flat on the bed all the way out, so there is no overhang to support.
    profile = CrossSection(
        [[
            (half - 4, tray_bot),
            (shoulder, tray_bot),
            (shoulder, 5),
            (shoulder - POST_L, 5),
            (shoulder - POST_L, tray_bot + ARM_STRIP),
            (half + 5, tray_bot + ARM_STRIP),
            (half, tray_top),
            (half - 4, tray_top),
        ]]
    )
    # Polygon (u, v) -> (y, z), extruded across X.
    web = profile.extrude(ARM_W).rotate((90, 0, 90)).translate((-ARM_W / 2, 0, 0))
    block = profile.extrude(BOSS_W).rotate((90, 0, 90)).translate((-BOSS_W / 2, 0, 0))
    block ^= box(-9, 9, shoulder - POST_L, shoulder + 1, -99, 99)
    nub = cyl_y(NUB_D, shoulder - 0.1, shoulder + NUB_L)
    screw = cyl_y(PIVOT_HOLE_D, shoulder - PIVOT_HOLE_DEPTH, shoulder + NUB_L + 1)
    screw += nut_slot_y(shoulder - NUT_WALL)
    return tray + both_ends(web + block + nub) - both_ends(screw)


def sensor_mount():
    """Plate under the tray for a small sensor board, stood off on two legs.

    M3 x 16s come down through the tray and legs into nuts in the plate. The
    board goes on with an M3 in a slot, or on foam tape.
    """
    top = -CG_ABOVE_TRAY - 3
    bottom = top - SENSOR_LEG - SENSOR_T
    half = SENSOR_W / 2
    length = SENSOR_Y + 4.5
    plate = box(-half, half, -length, length, bottom, bottom + SENSOR_T)
    for y in (-7.0, 7.0):
        plate -= box(-10, 10, y - M3_CLEAR_D / 2, y + M3_CLEAR_D / 2, bottom - 1, top)
    plate -= box(-2, 2, length - 2, length + 1, bottom - 1, top)  # marks the +Y end
    for y in (-SENSOR_Y, SENSOR_Y):
        plate += cyl_z(9, bottom, top, y=y)
        plate -= cyl_z(M3_CLEAR_D, bottom - 1, top + 1, y=y)
        plate -= hex_z(bottom - 1, bottom + NUT_RECESS).translate((0, y, 0))
    return plate


def _dome(radius, height):
    """Egg-shaped cap from the rim up, long axis on Y, flat on top at the given height."""
    centre = EGG_H / 2 - EGG_DIP - CG_ABOVE_TRAY
    c = height - centre
    rim = cover_top() - COVER_H
    m = math.tan(math.radians(COVER_OVERHANG))
    # Follow the ellipse until it gets too flat to print, then run straight to the top.
    r_t = radius**2 * m / math.hypot(c, radius * m)
    z_t = c**2 / math.hypot(c, radius * m)
    points = [(0.0, rim), (radius * math.sqrt(1 - ((rim - centre) / c) ** 2), rim)]
    a0 = math.asin((rim - centre) / c)
    a1 = math.asin(z_t / c)
    for i in range(1, 41):
        a = a0 + (a1 - a0) * i / 40
        points.append((radius * math.cos(a), centre + c * math.sin(a)))
    points += [(r_t - (height - centre - z_t) / m, height), (0.0, height)]
    return CrossSection([points]).revolve(96)


def cover_top():
    return EGG_H - EGG_DIP - CG_ABOVE_TRAY + COVER_FOAM + COVER_WALL


def cover():
    """Thin dome over the top of the egg. The drone's arms pass under it in notches."""
    top = cover_top()
    rim = top - COVER_H
    long_in, wide_in = EGG_L / 2 + COVER_CLEAR, EGG_W / 2 + COVER_CLEAR
    long_out, wide_out = long_in + COVER_WALL, wide_in + COVER_WALL
    hollow = _dome(long_in, top - COVER_WALL).scale((wide_in / long_in, 1, 1))
    shell = _dome(long_out, top).scale((wide_out / long_out, 1, 1))

    # Two ribs across the crown hold the clip on the centreline.
    gap = CLIP_W / 2 + 0.3
    for y in (-gap - 1.5, gap):
        shell += box(-wide_out, wide_out, y, y + 1.5, rim, top + 1.5) ^ (
            _dome(long_out + 1.5, top + 1.5).scale((wide_out / long_out, 1, 1))
        )
    part = shell - hollow
    notch = box(-ARM_NOTCH_W / 2, ARM_NOTCH_W / 2, 0, long_out + 5, rim - 1, rim + ARM_NOTCH_H)
    for angle in (45, 135, 225, 315):
        part -= notch.rotate((0, 0, angle))
    return part - cyl_z(CENTRE_HOLE_D, top - COVER_WALL - 1, top + 2)


def clip():
    """Spring band over the cover, notched to snap under both edges of the tray.

    It prints on its side so the layers run along the band, which is what lets
    the legs spread again and again. Pull the tabs apart to take it off.
    """
    tray_bot = -CG_ABOVE_TRAY - 3
    under = cover_top()
    floor = TRAY_HALF + 0.2
    crest = floor - CLIP_TOOTH
    r = 8.0

    def corner(radius, i):
        a = math.pi / 2 * i / 12
        return floor - r + radius * math.sin(a), under - r + radius * math.cos(a)

    inside = [(-1.0, under)] + [corner(r, i) for i in range(13)]
    for i in range(-CLIP_NOTCHES_SHORTER, CLIP_NOTCHES_LONGER + 1):
        flank = tray_bot - i * CLIP_PITCH
        inside += [(floor, flank), (crest, flank), (floor, flank - CLIP_RAMP)]
    foot = tray_bot - CLIP_NOTCHES_LONGER * CLIP_PITCH - CLIP_RAMP - 1
    tab = floor + CLIP_T + CLIP_TAB
    inside += [(floor, foot), (tab, foot)]
    outside = [(tab, foot + CLIP_T), (floor + CLIP_T, foot + CLIP_T)]
    outside += [corner(r + CLIP_T, i) for i in range(12, -1, -1)] + [(-1.0, under + CLIP_T)]
    points = inside + outside

    half = CrossSection([points]).extrude(CLIP_W).rotate((90, 0, 0)).translate((0, CLIP_W / 2, 0))
    return half + half.rotate((0, 0, 180)) - cyl_z(CENTRE_HOLE_D, under - 1, under + CLIP_T + 1)


def _frame_bearing_cut(face_z, outward):
    """Bearing through the bar's web at x=y=0; key access from outside the rig."""
    web_end = face_z + outward * PAD_T
    far = face_z + outward * (BAR + 1)
    seat = cyl_z(BEARING_D, *sorted((face_z - outward, web_end + outward)))
    cbore = cyl_z(CBORE_D, *sorted((web_end, far)))
    return seat + cbore


def _frame_end_cut(x, face_z, outward):
    """Socket for the upright, plus the M3's hole and head recess."""
    s = BAR / 2 + SOCKET_CLEAR / 2
    socket = box(x - s, x + s, -s, s, *sorted((face_z - outward, face_z + outward * SOCKET)))
    back = face_z + outward * BAR
    hole = cyl_z(M3_CLEAR_D, *sorted((face_z, back + outward)), x=x)
    head = cyl_z(M3_HEAD_D, *sorted((back - outward * M3_HEAD_H, back + outward)), x=x)
    return socket + hole + head


def top_bar():
    z0, z1 = FRAME_Z, FRAME_Z + BAR
    w = FRAME_W / 2
    bar = box(-w, w, -BAR / 2, BAR / 2, z0, z1) + cyl_z(FRAME_PAD_D, z0, z1)
    if not has_slip_ring():
        bar -= _frame_bearing_cut(z0, 1)
    else:
        bar -= cyl_z(CBORE_D, z0 - 1, z1 + 1)  # slip ring tube passes through, no bearing here
        for x in (-SLIP_FOOT_X, SLIP_FOOT_X):
            bar -= cyl_z(M3_THREAD_D, z1 - SLIP_FOOT_PILOT, z1 + 1, x=x)
    for x in (-UPRIGHT_X, UPRIGHT_X):
        bar -= _frame_end_cut(x, z0, 1)
    return bar


def _slip_flange_z():
    return FRAME_Z + BAR + SLIP_FOOT_T + SLIP_WINDOW + SLIP_PLATE_T


def slip_mount():
    """Stands on the top bar, spigot in the old counterbore; holds the slip ring's flange."""
    foot = FRAME_Z + BAR
    top = foot + SLIP_FOOT_T
    plate_z = top + SLIP_WINDOW
    flange_z = _slip_flange_z()
    half = BAR / 2

    part = box(-SLIP_FOOT_X - 6, SLIP_FOOT_X + 6, -half, half, foot, top)
    part += cyl_z(SLIP_SPIGOT_D, foot - SLIP_SPIGOT_L, foot + 0.1)
    wall = CrossSection(
        [[(12, top - 0.1), (SLIP_FOOT_X + 6, top - 0.1), (SLIP_FOOT_X - 4, plate_z + 0.1), (12, plate_z + 0.1)]]
    )
    wall = wall.extrude(BAR).rotate((90, 0, 0)).translate((0, half, 0))
    part += both_ends(wall)
    part += cyl_z(SLIP_PLATE_D, plate_z, flange_z)

    if has_slip_bearing():
        part -= cyl_z(SLIP_SPIGOT_BORE, foot - SLIP_SPIGOT_L - 1, foot + 0.1)
        part -= cyl_z(SLIP_SHELF_BORE, foot, top + 1)
        part -= cyl_z(SLIP_BEARING_D, top - SLIP_BEARING_W, top + 1)
    else:
        part -= cyl_z(SLIP_SPIGOT_BORE, foot - SLIP_SPIGOT_L - 1, top + 1)
    part -= cyl_z(SLIP_BODY_D + 0.6, plate_z - 1, flange_z + 1)
    for angle in (0, 120, 240):
        a = math.radians(angle)
        x, y = SLIP_PCD / 2 * math.cos(a), SLIP_PCD / 2 * math.sin(a)
        part -= cyl_z(M3_THREAD_D, flange_z - SLIP_PLATE_T - 1, flange_z + 1, x=x, y=y)
    for x in (-SLIP_FOOT_X, SLIP_FOOT_X):
        part -= cyl_z(M3_CLEAR_D, foot - 1, top + 1, x=x)
        part -= cyl_z(M3_HEAD_D, foot + SLIP_HEAD_SEAT, flange_z + 1, x=x)
    return part


def slip_bearing():
    top = FRAME_Z + BAR + SLIP_FOOT_T
    race = cyl_z(SLIP_BEARING_OD, top - SLIP_BEARING_W, top)
    return race - cyl_z(SLIP_TUBE_D, top - SLIP_BEARING_W - 1, top + 1)


def slip_tube():
    """Turns with the outer ring: boss below, the slip ring's stem in the top end."""
    bottom = FRAME_Z - NUB_L - 0.1
    flange_z = _slip_flange_z()
    stem_tip = flange_z - SLIP_SHOULDER_L - SLIP_STEM_L
    top = flange_z - SLIP_SHOULDER_L - SLIP_TUBE_GAP
    r = SLIP_TUBE_D / 2

    part = cyl_z(SLIP_TUBE_D, bottom, top)
    part += cyl_z(SLIP_SPIGOT_TUBE_D, SLIP_SPIGOT_END, bottom + 0.1)

    part -= cyl_z(SLIP_STEM_D + SLIP_STEM_FIT, stem_tip - 0.5, top + 1)
    part -= cyl_z(SLIP_BORE_D, bottom + SLIP_FLOOR, stem_tip)
    part -= cyl_z(SLIP_WIRE_BORE, SLIP_SPIGOT_END - 1, bottom + SLIP_FLOOR + 1)

    tip = SLIP_SPIGOT_TUBE_D / 2 - SLIP_LOCK_DIMPLE
    dimple = cyl_y(3.0, tip, r).rotate((0, 0, -90)).translate((0, 0, SLIP_LOCK_Z))
    part -= both_ends(dimple)
    screw = cyl_y(M3_THREAD_D, 0, r + 1).translate((0, 0, stem_tip + 4.5))
    return part - both_ends(screw)


def slip_ring_model():
    z = _slip_flange_z()
    ring = cyl_z(SLIP_STEM_D, z - SLIP_SHOULDER_L - SLIP_STEM_L, z - SLIP_SHOULDER_L + 0.1)
    ring += cyl_z(SLIP_BODY_D, z - SLIP_SHOULDER_L, z + SLIP_L - SLIP_SHOULDER_L)
    ring += cyl_z(SLIP_FLANGE_D, z, z + SLIP_FLANGE_T)
    return ring


def base():
    z1, z0 = -FRAME_Z, -FRAME_Z - BAR
    w = FRAME_W / 2
    plate = box(-w, w, -w, w, z0, z1) - box(-w + BAR, w - BAR, -w + BAR, w - BAR, z0 - 1, z1 + 1)
    plate += box(-w, w, -BAR / 2, BAR / 2, z0, z1)
    plate += cyl_z(FRAME_PAD_D, z0, z1)

    gusset = CrossSection([[(w, w), (w - 44, w), (w, w - 44)]]).extrude(BAR).translate((0, 0, z0))
    for angle in (0, 90, 180, 270):
        plate += gusset.rotate((0, 0, angle))

    r0 = BAR / 2 + SOCKET_CLEAR / 2
    rib = CrossSection([[(r0, 0), (r0 + RIB_L, 0), (r0 + 2, RIB_L - 2), (r0, RIB_L)]])
    rib = rib.extrude(RIB_T).rotate((90, 0, 90)).translate((-RIB_T / 2, 0, z1 - 0.01))
    for x in (-UPRIGHT_X, UPRIGHT_X):
        plate += both_ends(rib.translate((x, 0, 0)))

    plate -= _frame_bearing_cut(z1, -1)
    for x in (-UPRIGHT_X, UPRIGHT_X):
        plate -= _frame_end_cut(x, z1, -1)
    inset = w - 12
    for x, y in itertools.product((-inset, inset), repeat=2):  # M3 to the bench
        plate -= cyl_z(M3_CLEAR_D, z0 - 1, z1 + 1, x=x, y=y)
    return plate


def upright():
    """Rig orientation, standing on the base; rotate 90 degrees to print."""
    half = BAR / 2
    c = 2.0
    section = CrossSection(
        [[
            (half - c, -half), (half, -half + c), (half, half - c), (half - c, half),
            (-half + c, half), (-half, half - c), (-half, -half + c), (-half + c, -half),
        ]]
    )
    body = section.extrude(UPRIGHT_L).translate((0, 0, -UPRIGHT_L / 2))
    for z in (-UPRIGHT_L / 2 - 1, UPRIGHT_L / 2 - 14):
        body -= cyl_z(M3_THREAD_D, z, z + 15)
    return body


def bearing_fit_test():
    """Strip of bearing seats, printed on edge like the rings: 1-4 notches = 6.0-6.3."""
    sizes = (6.0, 6.1, 6.2, 6.3)
    pitch = 12.0
    length = pitch * len(sizes) + 3
    h = RING_H
    strip = box(0, length, 0, PAD_T, 0, h) + box(0, length, -3, PAD_T + 3, 0, 1)
    for i, d in enumerate(sizes):
        x = 7 + i * pitch
        strip -= teardrop_y(d, -1, PAD_T + 1).translate((x, 0, h / 2))
        for n in range(i + 1):
            strip -= box(x - 4 + n * 2, x - 3 + n * 2, -1, PAD_T + 1, h - 1, h + 1)
    return strip


def pivot_shims():
    """Eight in a row: six plus spares. Print with a brim."""
    shim = cyl_z(SHIM_OD, 0, SHIM_T) - cyl_z(SHIM_ID, -1, SHIM_T + 1)
    row = shim
    for i in range(1, 8):
        row += shim.translate((i * 7.0, 0, 0))
    return row


def base_sections():
    """Four corners: the front, rear and middle rails cut on X, the sides on Y."""
    w = FRAME_W / 2
    z_mid = -FRAME_Z - BAR / 2
    left, right = lap_split(base(), BASE_CUT, (-w + BAR / 2, 0, w - BAR / 2), z_mid)
    out = []
    for piece, side in ((left, -1), (right, 1)):
        rail = side * (w - BAR / 2)
        turned_piece = piece.rotate((0, 0, 90))  # side rail now runs along X, at y = rail
        out.extend(lap_split(turned_piece, BASE_CUT, (rail,), z_mid))
    return out


def print_ready(part, flip=None):
    """Move a part from rig pose to the bed: lowest point on z=0, centred in XY."""
    if flip is not None:
        part = part.rotate(flip)
    x0, y0, z0, x1, y1, z1 = part.bounding_box()
    return part.translate((-(x0 + x1) / 2, -(y0 + y1) / 2, -z0))


def parts_for_print():
    top_a, top_b = lap_split(top_bar(), TOP_CUT, (0,), FRAME_Z + BAR / 2)
    parts = {
        "upright": print_ready(upright(), (0, 90, 0)),
        "top_bar_a": print_ready(top_a),
        "top_bar_b": print_ready(top_b),
        "outer_ring": print_ready(outer_ring()),
        "inner_ring": print_ready(inner_ring()),
        "cradle": print_ready(cradle()),
        "sensor_mount": print_ready(sensor_mount()),
        "cover": print_ready(cover()),
        "clip": print_ready(clip(), (90, 0, 0)),
        "bearing_fit_test": print_ready(bearing_fit_test()),
        "pivot_shims": print_ready(pivot_shims()),
    }
    if has_slip_ring():
        parts["slip_mount"] = print_ready(slip_mount(), (180, 0, 0))
        parts["slip_tube"] = print_ready(slip_tube(), (180, 0, 0))
    for i, section in enumerate(base_sections()):
        parts[f"base_{'abcd'[i]}"] = print_ready(section)
    return parts


def _at_motors(part):
    m = MOTOR_SPACING / 2
    out = None
    for sx, sy in itertools.product((-1, 1), repeat=2):
        piece = part.translate((sx * m, sy * m, 0))
        out = piece if out is None else out + piece
    return out


def drone_model():
    """Stand-in for the preview: egg, four arm rods, motors and prop discs."""
    egg = Manifold.sphere(1, 48).scale((EGG_W / 2, EGG_L / 2, EGG_H / 2))
    egg = egg.translate((0, 0, EGG_H / 2 - EGG_DIP - CG_ABOVE_TRAY))
    reach = math.hypot(MOTOR_SPACING, MOTOR_SPACING) / 2
    rod = box(-2, 2, 0, reach, *ARM_Z)
    rods = None
    for angle in (45, 135, 225, 315):
        piece = rod.rotate((0, 0, angle))
        rods = piece if rods is None else rods + piece
    motors = _at_motors(cyl_z(11, MOTOR_BOTTOM, PROP_Z))
    props = _at_motors(cyl_z(PROP_D, PROP_Z, PROP_Z + 1))
    return egg + rods + motors + props


def drone_envelope():
    """What must stay clear: the prop sweep, motors and body, with some spare."""
    props = _at_motors(cyl_z(DRONE - MOTOR_SPACING, PROP_Z - 1, PROP_TOP))
    props += _at_motors(cyl_z(14, MOTOR_BOTTOM, PROP_Z - 1))
    h = (EGG_W + 4) / 2, (EGG_L + 4) / 2
    return props + box(-h[0], h[0], -h[1], h[1], -CG_ABOVE_TRAY, DRONE_H - CG_ABOVE_TRAY)


def heads_on_y(a):
    """Screw heads for a pair of bearings on +-Y of a seat with inner face at a."""
    seat = a + PAD_T + 0.1
    return both_ends(cyl_y(HEAD_D, seat + SHIM_T, seat + SHIM_T + HEAD_H))


def bearings_on_y(a):
    r = cyl_y(6.0, a - 0.1, a - 0.1 + BEARING_W) - cyl_y(3.0, a - 1, a + BEARING_W + 1)
    return both_ends(r)


def rot(axis, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def turned(part, matrix):
    m = np.hstack([matrix, np.zeros((3, 1))])
    return part.transform(m.tolist())


def assembly_groups():
    """Moving groups in rig pose, zeroed. Each screw head turns with its boss."""
    cr = cradle()
    inner = inner_ring()
    # Outer ring: print flat -> ring axis on rig Y, bosses on Z, bearings on X.
    outer = outer_ring().rotate((90, 0, 0))
    frame = base() + top_bar()
    if has_slip_ring():
        frame += slip_mount()
    for x in (-UPRIGHT_X, UPRIGHT_X):
        frame += upright().translate((x, 0, 0))

    h_cr = heads_on_y(INNER_Y)
    h_inner = heads_on_y(OUTER_X).rotate((0, 0, 90))
    if has_slip_ring():
        below = box(-99, 99, -99, 99, -999, 0)  # the top pivot is now the slip ring
        lock_head = cyl_y(HEAD_D, OUTER_BOSS_W / 2, OUTER_BOSS_W / 2 + HEAD_H)
        lock_head = lock_head.rotate((0, 0, -90)).translate((0, 0, SLIP_LOCK_Z))
        h_outer = heads_on_y(FRAME_Z).rotate((90, 0, 0)) ^ below
        outer = outer + h_outer + slip_tube() + lock_head
    else:
        outer = outer + heads_on_y(FRAME_Z).rotate((90, 0, 0))
    inner += h_inner
    cr = cr + sensor_mount() + cover() + clip() + h_cr
    b_frame = bearings_on_y(FRAME_Z).rotate((90, 0, 0))
    if has_slip_ring():
        b_frame = b_frame ^ below
    if has_slip_bearing():
        b_frame += slip_bearing()

    return {
        "frame": frame,
        "outer": outer,
        "inner": inner,
        "cradle": cr,
        "bearings": {
            "frame": b_frame,
            "outer": bearings_on_y(OUTER_X).rotate((0, 0, 90)),
            "inner": bearings_on_y(INNER_Y),
        },
    }


def check():
    g = assembly_groups()
    frame = g["frame"]
    outer = g["outer"]
    inner = g["inner"]
    moving_drone = g["cradle"] + drone_envelope()
    cr = cradle() + sensor_mount()

    print(f"inner {2 * INNER_X} x {2 * INNER_Y:.0f}, outer {2 * OUTER_X:.0f} x {2 * OUTER_Z:.0f}")
    print(f"cradle arms into drone: {(cr ^ drone_envelope()).volume():.1f} mm3")
    assert clip().volume() > 0 and cover().volume() > 0
    egg = drone_model()
    print(f"cover into drone: {(cover() ^ egg).volume():.1f} mm3, gap {cover().min_gap(egg, 40.0):.2f} mm")
    print(f"clip into drone: {(clip() ^ egg).volume():.1f} mm3, gap {clip().min_gap(egg, 40.0):.2f} mm")
    print(f"clip into cover: {(clip() ^ cover()).volume():.2f} mm3, gap {clip().min_gap(cover(), 40.0):.2f} mm")
    print(f"clip into cradle: {(clip() ^ cradle()).volume():.2f} mm3, gap {clip().min_gap(cradle(), 40.0):.2f} mm")
    print(f"cover into cradle: {(cover() ^ cradle()).volume():.2f} mm3, gap {cover().min_gap(cradle(), 40.0):.2f} mm")
    print(f"frame {FRAME_W:.0f} wide x {2 * (FRAME_Z + BAR):.0f} tall, upright {UPRIGHT_L:.0f}")

    worst = {}
    collisions = 0

    def test(name, a, b):
        nonlocal collisions
        overlap = (a ^ b).volume()
        gap = 0.0 if overlap > 1e-6 else a.min_gap(b, 40.0)
        if overlap > 1e-6:
            collisions += 1
            print(f"  COLLISION {name}: {overlap:.3f} mm3")
        worst[name] = min(worst.get(name, 9e9), gap)

    angles = list(range(0, 360, 15))
    for yaw in angles:
        y = rot("z", yaw)
        test("frame / outer ring", frame, turned(outer, y))
    for pitch in angles:
        p = rot("x", pitch)
        test("outer ring / inner ring", outer, turned(inner, p))
    for roll in angles:
        r = rot("y", roll)
        test("inner ring / cradle + drone box", inner, turned(moving_drone, r))
    for pitch, roll in itertools.product(angles[::2], repeat=2):
        m = rot("x", pitch) @ rot("y", roll)
        test("outer ring / cradle + drone box", outer, turned(moving_drone, m))
    for yaw, pitch, roll in itertools.product(angles[::4], repeat=3):
        m = rot("z", yaw) @ rot("x", pitch) @ rot("y", roll)
        test("frame / cradle + drone box", frame, turned(moving_drone, m))
    for yaw, pitch in itertools.product(angles[::2], repeat=2):
        m = rot("z", yaw) @ rot("x", pitch)
        test("frame / inner ring", frame, turned(inner, m))

    props = drone_envelope()
    for roll in angles:
        test("PROP CLEARANCE inner ring", inner, turned(props, rot("y", roll)))
    for pitch, roll in itertools.product(angles[::2], repeat=2):
        test("PROP CLEARANCE outer ring", outer, turned(props, rot("x", pitch) @ rot("y", roll)))
    for yaw, pitch, roll in itertools.product(angles[::4], repeat=3):
        m = rot("z", yaw) @ rot("x", pitch) @ rot("y", roll)
        test("PROP CLEARANCE frame", frame, turned(props, m))

    for name, gap in worst.items():
        print(f"  {name}: min gap {gap:.2f} mm")
    print("OK: all three axes turn fully" if not collisions else f"{collisions} COLLISIONS")
    return collisions


def mesh_arrays(part):
    mesh = part.to_mesh()
    return np.asarray(mesh.vert_properties, np.float32)[:, :3], np.asarray(mesh.tri_verts, np.uint32)


def export_stl(versioned=False):
    """Every part into stl/ and 3mf/, or with versioned into v<VERSION>/stl and v<VERSION>/3mf."""
    import trimesh

    if versioned:
        root, suffix = OUT / f"v{VERSION}", f"_v{VERSION}"
    else:
        root, suffix = OUT, ""
    stl_dir, mf_dir = root / "stl", root / "3mf"
    stl_dir.mkdir(parents=True, exist_ok=True)
    mf_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, part in parts_for_print().items():
        v, f = mesh_arrays(part)
        tm = trimesh.Trimesh(v, f, process=False)
        assert tm.is_watertight, name
        assert name == "pivot_shims" or tm.body_count == 1, name
        tm.export(stl_dir / f"{name}{suffix}.stl")
        tm.export(mf_dir / f"{name}{suffix}.3mf")
        size = tm.extents
        assert sorted(size)[1] <= BED and size.max() <= BED, name
        grams = tm.volume * PLA_DENSITY
        print(f"{name:18s} {size[0]:6.1f} x {size[1]:6.1f} x {size[2]:5.1f} mm  "
              f"{tm.volume / 1000:6.1f} cm3  solid {grams:5.1f} g")
        rows.append((name, size, grams))
    return rows


PLA_DENSITY = 1.24e-3  # g/mm3
PRINT_COUNT = {"upright": 2}
PRINT_NOTES = {
    "base_a": "Base frame, corner section", "base_b": "Base frame, corner section",
    "base_c": "Base frame, corner section", "base_d": "Base frame, corner section",
    "upright": "Frame sides", "top_bar_a": "Top bar, with the top pivot", "top_bar_b": "Top bar",
    "outer_ring": "Yaw ring", "inner_ring": "Pitch ring", "cradle": "Roll axis, holds the drone",
    "sensor_mount": "Sensor board plate under the tray", "cover": "Holds the drone body down",
    "clip": "Snaps the cover on", "slip_mount": "Holds the slip ring on the top bar",
    "slip_tube": "Turns with the outer ring, carries the slip ring stem",
    "bearing_fit_test": "Print first, to tune BEARING_D", "pivot_shims": "8 on one plate, 6 needed. Use a brim",
}


def hardware():
    """(part, qty, used for) for the current version."""
    slip = has_slip_ring()
    pivots = 5 if slip else 6  # from v1.1 the slip ring is the top pivot
    rows = [
        ("MR63ZZ bearing, 3 × 6 × 2.5 mm", pivots, "Two per pivot" + (", none at the top pivot" if slip else "")),
        ("M3 × 8 socket head screw", pivots + 3 * slip,
         f"{pivots} pivot axles" + (", 3 slip ring flange" if slip else "")),
        ("M3 × 12 socket head screw", 4, "Uprights to base and top bar"),
        ("M3 × 16 socket head screw", 2 + 2 * slip, "2 sensor mount to tray" + (", 2 slip mount to top bar" if slip else "")),
        ("M3 × 20 socket head screw", 12, "Frame lap joints"),
        ("M3 hex nut, 5.5 mm across flats", pivots + 14, f"{pivots} pivots, 12 frame joints, 2 sensor mount"),
    ]
    if slip:
        rows += [
            ("M3 × 5 socket head screw", 1, "Side screw, outer ring boss to slip tube"),
            ("M3 × 6 set screw, or short M3 screw", 2, "Slip tube onto the slip ring stem"),
            ("M3 washer, 7 mm OD", 3, "Under the slip ring flange screws"),
            ("Slip ring, 12 wire, 22 mm body (SNM022A-12)", 1, "Top pivot, and power to the drone"),
        ]
    if has_slip_bearing():
        rows.append(("6702ZZ ball bearing, 15 × 21 × 4 mm", 1, "Steadies the slip tube inside slip_mount"))
    rows += [
        ("M3 screw or 3 mm woodscrew, 30 mm or longer", 4, "Base to the bench"),
        ("Soft foam pad, about 15 × 20 × 2–3 mm", 1, "Inside the crown of the cover"),
    ]
    return rows


def export_bom(rows):
    """v<VERSION>/BOM_v<VERSION>.md and .csv: printed parts and hardware."""
    import csv

    folder = OUT / f"v{VERSION}"
    printed = [
        (f"{name}_v{VERSION}", PRINT_COUNT.get(name, 1), " × ".join(f"{s:.0f}" for s in size),
         grams * PRINT_COUNT.get(name, 1), PRINT_NOTES[name])
        for name, size, grams in rows
    ]
    total = sum(p[3] for p in printed)
    hw = hardware()

    md = [f"# Mini Drone Gimbal v{VERSION}: bill of materials", ""]
    md += ["Generated by `mini_drone_gimbal.py`. Print files are in `stl/` and `3mf/`.", ""]
    md += ["## Printed parts", "", "| File | Qty | Size, mm | Solid mass | Notes |", "|---|---|---|---|---|"]
    md += [f"| {f} | {q} | {s} | {g:.0f} g | {n} |" for f, q, s, g, n in printed]
    md += ["", f"Total solid mass {total / 1000:.2f} kg. That is the upper limit: with 15% infill on the "
           "frame it prints much lighter. PLA or PETG, 0.2 mm layers, no supports.", ""]
    md += ["## Hardware", "", "| Part | Qty | Used for |", "|---|---|---|"]
    md += [f"| {p} | {q} | {u} |" for p, q, u in hw]
    (folder / f"BOM_v{VERSION}.md").write_text("\n".join(md) + "\n")

    with open(folder / f"BOM_v{VERSION}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["type", "item", "qty", "size_mm", "solid_g", "notes"])
        for f, q, s, g, n in printed:
            w.writerow(["printed", f, q, s, f"{g:.1f}", n])
        for p, q, u in hw:
            w.writerow(["hardware", p, q, "", "", u])
    print(f"wrote v{VERSION}/BOM_v{VERSION}.md and .csv")


def export_html(path=None):
    g = assembly_groups()
    def pack(part):
        v, f = mesh_arrays(part)
        return {
            "v": base64.b64encode(np.round(v, 2).astype(np.float32).tobytes()).decode(),
            "f": base64.b64encode(f.astype(np.uint32).tobytes()).decode(),
        }

    def sign(part, i):
        b = part.bounding_box()
        c = (b[i] + b[i + 3]) / 2
        return 0 if abs(c) < 1 else math.copysign(1, c)

    w = FRAME_W / 2
    frame = [(b.rotate((0, 0, -90)), -1) for b in base_sections()]
    frame += [(t, 1) for t in lap_split(top_bar(), TOP_CUT, (0,), FRAME_Z + BAR / 2)]
    frame += [(upright().translate((x, 0, 0)), 0) for x in (-UPRIGHT_X, UPRIGHT_X)]
    if has_slip_ring():
        frame += [(slip_mount(), 1)]

    data = {
        "frameParts": [
            {**pack(m), "dir": [sign(m, 0), sign(m, 1) if up < 0 else 0, up]} for m, up in frame
        ],
        "outer": pack(g["outer"]),
        "inner": pack(g["inner"]),
        "cradle": pack(cradle() + sensor_mount() + heads_on_y(INNER_Y)),
        "slipRing": pack(slip_ring_model() if has_slip_ring() else Manifold()),
        "cover": pack(cover()),
        "clip": pack(clip()),
        "drone": pack(drone_model()),
        "envelope": pack(drone_envelope()),
        "bearingsFrame": pack(g["bearings"]["frame"]),
        "bearingsOuter": pack(g["bearings"]["outer"]),
        "bearingsInner": pack(g["bearings"]["inner"]),
        "frameW": FRAME_W,
        "frameH": 2 * (FRAME_Z + BAR),
    }
    template = (OUT / "preview_template.html").read_text()
    html = template.replace("/*MODEL*/null", json.dumps(data))
    html = html.replace(">v1.2</small>", f">v{VERSION}</small>")
    path = path or OUT / "preview.html"
    path.write_text(html)
    return path


def export_png():
    """v<VERSION>/preview_v<VERSION>.png: headless Chrome shoots the 3D view of a temporary preview."""
    import subprocess
    import tempfile

    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    png = OUT / f"v{VERSION}" / f"preview_v{VERSION}.png"
    with tempfile.TemporaryDirectory() as tmp:
        page = export_html(Path(tmp) / "preview.html")
        subprocess.run(
            [chrome, "--headless", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
             "--hide-scrollbars", "--window-size=1100,760", "--virtual-time-budget=15000",
             f"--screenshot={png}", page.as_uri() + "#shot"],
            check=True, capture_output=True,
        )
    print(f"wrote {png.relative_to(OUT)}")


def export_version(version):
    """Check one version, then write all its parts and a preview image into v<version>/."""
    set_version(version)
    print(f"--- v{version}")
    set_circular_segments(32)
    if check():
        sys.exit(1)
    set_circular_segments(64)
    export_bom(export_stl(versioned=True))
    export_png()


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "versions":
        for v in VERSIONS:
            export_version(v)
        sys.exit(0)
    if what.lstrip("v") in VERSIONS:
        export_version(what.lstrip("v"))
        sys.exit(0)
    set_circular_segments(32 if what == "check" else 64)
    if what in ("check", "all") and check():
        sys.exit(1)
    if what in ("stl", "all"):
        export_stl()
    if what in ("html", "all"):
        export_html()
