# Standalone-Newton port of the robot-hand taxel sensor (from IsaacLab demos
# _hand_taxels.py). Same physics: rasterize MJWarp per-contact data into per-pad
# taxel grids, decoding forces with mujoco_warp's own contact_force_fn. The ONLY
# change vs the Isaac version is the source of pad geometry: instead of opening the
# USD stage (Kit-only), pad specs are derived from the FlexHand's finger collision
# meshes (body-local verts we stashed at build time). Reads solver.mjw_data, which
# standalone newton.solvers.SolverMuJoCo exposes.
#
# NOTE (warp JIT): needs the matched NVRTC on the loader path, like the original:
#   LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

try:
    from mujoco_warp._src.support import contact_force_fn
    from mujoco_warp._src.types import vec5
except ImportError as exc:  # pragma: no cover
    raise ImportError("mujoco_warp private API moved (contact_force_fn / vec5); "
                      "pinned to mujoco_warp 3.8 semantics.") from exc

# pad name -> FlexHand finger-body name (the phalanx links). Verbatim from the
# original DIGIT_PAD_LINKS; these names match the USD body names FlexHand builds.
DIGIT_PAD_LINKS: dict[str, str] = {
    "ring.meta": "Metacarpal_Bone_V02_02", "ring.prox": "Proximal_Phalanx_Bone_V02",
    "ring.dist": "Distal_Phalanx_Bone_V02",
    "middle.meta": "Metacarpal_Bone_V02", "middle.prox": "Proximal_Phalanx_Bone_V02_middlefinger",
    "middle.dist": "Distal_Phalanx_Bone_V02_03",
    "index.meta": "Metacarpal_Bone_V02_01", "index.prox": "Proximal_Phalanx_Bone_V02_02",
    "index.dist": "Distal_Phalanx_Bone_V02_04",
    "pinky.meta": "Metacarpal_Bone_V02_pinky", "pinky.prox": "Proximal_Phalanx_Bone_V02_pinky",
    "pinky.dist": "Distal_Phalanx_Bone_V02_02",
    "thumb.meta": "Metacarpal_Bone_V02_03", "thumb.prox": "Proximal_Phalanx_Bone_V02_01",
    "thumb.dist": "Distal_Phalanx_Bone_V02_01",
}

CONTACT_TYPE_CONSTRAINT = wp.constant(1)


def _fail(msg):
    raise RuntimeError(f"[hand_taxels] {msg}")


@dataclass
class PadSpec:
    name: str
    link: str
    newton_body: int
    mj_body: int
    origin_b: np.ndarray
    rot_b: np.ndarray            # 3x3, ROWS = pad axes (u, v, outward normal) in body frame
    size: tuple
    hw: tuple
    pitch: float
    offset: int
    h_tol: float


class HandTaxelData:
    """Views over the sensor's flat GPU buffers (zero-copy via wp.to_torch)."""

    def __init__(self, sensor):
        import torch
        self._s = sensor
        flat_n = wp.to_torch(sensor._grid_fn)
        flat_s = wp.to_torch(sensor._grid_fs)
        self.tactile_normal_force = {}
        self.tactile_shear_force = {}
        for p in sensor.pads:
            h, w = p.hw
            self.tactile_normal_force[p.name] = flat_n[p.offset:p.offset + h * w].view(h, w)
            self.tactile_shear_force[p.name] = flat_s[p.offset:p.offset + h * w].view(h, w, 2)
        self._totals = wp.to_torch(sensor._pad_total)
        self._dropped = wp.to_torch(sensor._pad_dropped)

    @property
    def pad_force_total(self):    # (P,) binned normal force [N]
        return self._totals

    @property
    def pad_force_dropped(self):
        return self._dropped


@wp.func
def _mj_quat_to_xyzw(q: wp.quat) -> wp.quat:
    return wp.quat(q[1], q[2], q[3], q[0])


@wp.kernel
def _rasterize_contacts(
    con_dist: wp.array(dtype=float),
    con_pos: wp.array(dtype=wp.vec3),
    con_frame: wp.array(dtype=wp.mat33),
    con_friction: wp.array(dtype=vec5),
    con_dim: wp.array(dtype=int),
    con_geom: wp.array(dtype=wp.vec2i),
    con_efc_address: wp.array2d(dtype=int),
    con_worldid: wp.array(dtype=int),
    con_type: wp.array(dtype=int),
    nacon: wp.array(dtype=int),
    naconmax: int,
    efc_force: wp.array2d(dtype=float),
    njmax: int,
    opt_cone: int,
    geom_to_pad: wp.array(dtype=int),
    pad_body_mj: wp.array(dtype=int),
    pad_origin_b: wp.array(dtype=wp.vec3),
    pad_rot_b: wp.array(dtype=wp.mat33),
    pad_hw: wp.array(dtype=wp.vec2i),
    pad_offset: wp.array(dtype=int),
    pad_pitch: wp.array(dtype=float),
    pad_h_tol: wp.array(dtype=float),
    n_pads: int,
    total: int,
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    grid_fn: wp.array(dtype=float),
    grid_fs: wp.array(dtype=wp.vec2),
    pad_total: wp.array(dtype=float),
    pad_dropped: wp.array(dtype=float),
):
    cid = wp.tid()
    if cid >= wp.min(nacon[0], naconmax):
        return
    if (con_type[cid] & CONTACT_TYPE_CONSTRAINT) == 0:
        return
    if con_efc_address[cid, 0] < 0:
        return
    world = con_worldid[cid]
    pad_base = world * n_pads          # batched: per-world block in pad_total/pad_dropped
    f6 = contact_force_fn(opt_cone, con_frame, con_friction, con_dim, con_efc_address,
                          efc_force, njmax, nacon, world, cid, False)
    f_normal = f6[0]
    if f_normal <= 0.0:
        return
    frame = con_frame[cid]
    n_w = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])
    t1_w = wp.vec3(frame[1, 0], frame[1, 1], frame[1, 2])
    t2_w = wp.vec3(frame[2, 0], frame[2, 1], frame[2, 2])
    for side in range(2):
        g = con_geom[cid][side]
        if g < 0:
            continue
        pad = geom_to_pad[g]
        if pad < 0:
            continue
        sgn = 1.0
        if side == 0:
            sgn = -1.0
        p_w = con_pos[cid] + (sgn * 0.5 * con_dist[cid]) * n_w
        b = pad_body_mj[pad]
        x_wb = wp.transform(xpos[world, b], _mj_quat_to_xyzw(xquat[world, b]))
        p_b = wp.transform_point(wp.transform_inverse(x_wb), p_w)
        p_pad = pad_rot_b[pad] @ (p_b - pad_origin_b[pad])
        if wp.abs(p_pad[2]) > pad_h_tol[pad]:
            wp.atomic_add(pad_dropped, pad_base + pad, f_normal)
            continue
        f_shear_w = sgn * (f6[1] * t1_w + f6[2] * t2_w)
        f_shear_b = wp.quat_rotate_inv(wp.transform_get_rotation(x_wb), f_shear_w)
        f_shear_pad = pad_rot_b[pad] @ f_shear_b
        h = pad_hw[pad][0]
        w_ = pad_hw[pad][1]
        pitch = pad_pitch[pad]
        cu = p_pad[0] / pitch + 0.5 * float(h - 1)
        cv = p_pad[1] / pitch + 0.5 * float(w_ - 1)
        if cu < -0.5 or cu > float(h) - 0.5 or cv < -0.5 or cv > float(w_) - 0.5:
            wp.atomic_add(pad_dropped, pad_base + pad, f_normal)
            continue
        i0 = wp.clamp(int(wp.floor(cu)), 0, h - 1)
        j0 = wp.clamp(int(wp.floor(cv)), 0, w_ - 1)
        i1 = wp.min(i0 + 1, h - 1)
        j1 = wp.min(j0 + 1, w_ - 1)
        wu = wp.clamp(cu - wp.floor(cu), 0.0, 1.0)
        wv = wp.clamp(cv - wp.floor(cv), 0.0, 1.0)
        base = world * total + pad_offset[pad]
        s2 = wp.vec2(f_shear_pad[0], f_shear_pad[1])
        w00 = (1.0 - wu) * (1.0 - wv)
        w01 = (1.0 - wu) * wv
        w10 = wu * (1.0 - wv)
        w11 = wu * wv
        wp.atomic_add(grid_fn, base + i0 * w_ + j0, w00 * f_normal)
        wp.atomic_add(grid_fn, base + i0 * w_ + j1, w01 * f_normal)
        wp.atomic_add(grid_fn, base + i1 * w_ + j0, w10 * f_normal)
        wp.atomic_add(grid_fn, base + i1 * w_ + j1, w11 * f_normal)
        wp.atomic_add(grid_fs, base + i0 * w_ + j0, w00 * s2)
        wp.atomic_add(grid_fs, base + i0 * w_ + j1, w01 * s2)
        wp.atomic_add(grid_fs, base + i1 * w_ + j0, w10 * s2)
        wp.atomic_add(grid_fs, base + i1 * w_ + j1, w11 * s2)
        wp.atomic_add(pad_total, pad_base + pad, f_normal)


def _pad_spec_from_verts(name, link, newton_body, local_verts, pitch, offset, margin=1.5e-3):
    """Robust pad geometry from a finger body's local collision verts: pad plane =
    the thin (smallest-extent) axis; origin at the AABB CENTER so contacts on either
    face register; size = the two larger extents; h_tol = half-thickness + margin.
    This guarantees pad_force_total = total normal force on the phalanx (what the
    in-viewer heatmap reads), using the same rasterization kernel as the original."""
    v = np.asarray(local_verts, dtype=np.float64)
    lo, hi = v.min(0), v.max(0)
    ext = hi - lo
    center = 0.5 * (lo + hi)
    order = np.argsort(ext)            # [thin, mid, long]
    n_ax, u_ax, v_ax = order[0], order[2], order[1]   # normal=thin, u=long, v=mid
    eye = np.eye(3)
    rot = np.stack([eye[u_ax], eye[v_ax], eye[n_ax]]).astype(np.float64)  # rows u,v,n
    size_u, size_v = float(ext[u_ax]), float(ext[v_ax])
    hw = (max(2, int(math.ceil(size_u / pitch))), max(2, int(math.ceil(size_v / pitch))))
    h_tol = 0.5 * float(ext[n_ax]) + margin
    return PadSpec(name=name, link=link, newton_body=newton_body, mj_body=-1,
                   origin_b=center.astype(np.float32), rot_b=rot, size=(size_u, size_v),
                   hw=hw, pitch=pitch, offset=offset, h_tol=h_tol)


class HandTaxelSensor:
    """Per-pad taxel pressure maps from MJWarp contacts, for a FlexHand in a
    standalone SolverMuJoCo model. Pads = the 15 finger phalanges."""

    def __init__(self, hand, solver, model, *, pitch=1.5e-3):
        if not hasattr(solver, "mjw_data"):
            _fail(f"solver {type(solver).__name__} has no mjw_data (need SolverMuJoCo)")
        self.d = solver.mjw_data
        self.m = solver.mjw_model
        self.naconmax = int(self.d.naconmax)
        self.njmax = int(self.d.njmax)
        self.opt_cone = int(self.m.opt.cone)
        self.device = wp.get_device()

        # ---- pad specs from the hand's finger collision meshes ----
        self.pads = []
        offset = 0
        for name, link in DIGIT_PAD_LINKS.items():
            if link not in hand.finger_body or link not in hand.finger_local:
                continue
            p = _pad_spec_from_verts(name, link, hand.finger_body[link],
                                     hand.finger_local[link], pitch, offset)
            self.pads.append(p)
            offset += p.hw[0] * p.hw[1]
        if not self.pads:
            _fail("no finger pad bodies found on the hand")

        # ---- geom -> pad lookup (verbatim mechanism) ----
        g2s = solver.mjc_geom_to_newton_shape.numpy()      # [nworld, ngeom]
        shape_body = model.shape_body.numpy()
        geom_bodyid = self.m.geom_bodyid.numpy()
        body_to_pad = {p.newton_body: i for i, p in enumerate(self.pads)}
        ngeom = g2s.shape[1]
        geom_to_pad = np.full(ngeom, -1, dtype=np.int32)
        pad_mj_body = np.full(len(self.pads), -1, dtype=np.int32)
        for g in range(ngeom):
            s = int(g2s[0, g])
            if s < 0 or s >= shape_body.shape[0]:
                continue
            pad = body_to_pad.get(int(shape_body[s]), -1)
            if pad < 0:
                continue
            geom_to_pad[g] = pad
            pad_mj_body[pad] = int(geom_bodyid[g])
        for i, p in enumerate(self.pads):
            if pad_mj_body[i] < 0:
                _fail(f"pad {p.name}: no MuJoCo geom for link {p.link}")
            p.mj_body = int(pad_mj_body[i])
        n_mapped = int((geom_to_pad >= 0).sum())
        print(f"[hand_taxels] {len(self.pads)} pads, {n_mapped} sensorized geoms", flush=True)

        # ---- GPU tables + buffers ----
        # Batched: one SolverMuJoCo can step `world_count` homogeneous replicas. The
        # geom->pad table is world-LOCAL (geom ids 0..ngeom-1 per world, the same map
        # for every replica) and contacts carry con_worldid, so per-world force lands
        # in a per-world block of pad_total/grid (index = world*n_pads + pad). N=1 keeps
        # the original single-world behavior (world 0 -> offset 0).
        self.num_worlds = int(getattr(model, "world_count", 1) or 1)
        total = sum(p.hw[0] * p.hw[1] for p in self.pads)
        self.total_taxels = total
        self._geom_to_pad = wp.array(geom_to_pad, dtype=int, device=self.device)
        self._pad_body_mj = wp.array(pad_mj_body, dtype=int, device=self.device)
        self._pad_origin = wp.array(np.stack([p.origin_b for p in self.pads]).astype(np.float32),
                                    dtype=wp.vec3, device=self.device)
        self._pad_rot = wp.array(np.stack([p.rot_b for p in self.pads]).astype(np.float32),
                                 dtype=wp.mat33, device=self.device)
        self._pad_hw = wp.array(np.array([[p.hw[0], p.hw[1]] for p in self.pads], dtype=np.int32),
                                dtype=wp.vec2i, device=self.device)
        self._pad_offset = wp.array(np.array([p.offset for p in self.pads], dtype=np.int32),
                                    dtype=int, device=self.device)
        self._pad_pitch = wp.array(np.array([p.pitch for p in self.pads], dtype=np.float32),
                                   dtype=float, device=self.device)
        self._pad_h_tol = wp.array(np.array([p.h_tol for p in self.pads], dtype=np.float32),
                                   dtype=float, device=self.device)
        W = self.num_worlds
        self.n_pads = len(self.pads)
        self._grid_fn = wp.zeros(W * total, dtype=float, device=self.device)
        self._grid_fs = wp.zeros(W * total, dtype=wp.vec2, device=self.device)
        self._pad_total = wp.zeros(W * self.n_pads, dtype=float, device=self.device)
        self._pad_dropped = wp.zeros(W * self.n_pads, dtype=float, device=self.device)
        self._data = HandTaxelData(self)

    def update(self):
        self._grid_fn.zero_()
        self._grid_fs.zero_()
        self._pad_total.zero_()
        self._pad_dropped.zero_()
        wp.launch(
            _rasterize_contacts, dim=self.naconmax,
            inputs=[
                self.d.contact.dist, self.d.contact.pos, self.d.contact.frame,
                self.d.contact.friction, self.d.contact.dim, self.d.contact.geom,
                self.d.contact.efc_address, self.d.contact.worldid, self.d.contact.type,
                self.d.nacon, self.naconmax, self.d.efc.force, self.njmax, self.opt_cone,
                self._geom_to_pad, self._pad_body_mj, self._pad_origin, self._pad_rot,
                self._pad_hw, self._pad_offset, self._pad_pitch, self._pad_h_tol,
                self.n_pads, self.total_taxels,
                self.d.xpos, self.d.xquat,
            ],
            outputs=[self._grid_fn, self._grid_fs, self._pad_total, self._pad_dropped],
            device=self.device,
        )

    @property
    def data(self):
        return self._data

    def pad_totals_np(self):
        """Per-pad total normal force [N] as numpy (P,) for single-world, or
        (num_worlds*P,) batched, in self.pads order."""
        return self._pad_total.numpy()

    def pad_totals_torch(self):
        """Per-pad total normal force as a zero-copy torch view [num_worlds, P]."""
        return wp.to_torch(self._pad_total).view(self.num_worlds, self.n_pads)
