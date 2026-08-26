# Probe: small-bone geometry + finger joint graph, to plan the scale-up.
import os, numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

# Assets ship in ../assets of the DexiGrab repo; override with DEXIGRAB_ASSETS.
_ASSETS = os.environ.get("DEXIGRAB_ASSETS") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "assets")
U = os.path.join(_ASSETS, "robot_hand_flexpalm.usda")
stage = Usd.Stage.Open(U)
xc = UsdGeom.XformCache()

def centroid_extent(prim):
    # find a Mesh under prim
    m = None
    for p in Usd.PrimRange(prim):
        if p.GetTypeName() == "Mesh":
            m = p; break
    if m is None: return None
    pts = np.asarray(UsdGeom.Mesh(m).GetPointsAttr().Get(), dtype=np.float64)
    M = np.array(xc.GetLocalToWorldTransform(m), dtype=np.float64).reshape(4,4)
    w = (np.concatenate([pts, np.ones((len(pts),1))],1) @ M)[:,:3]
    return w.mean(0), w.min(0), w.max(0), len(pts)

PALM = "/Robotic_Hand_V5_simulacra/Palm_rigid"
print("===== BIG bones =====")
bigs = {}
for nm in ["Base_Bone_12_V02_pinky","Base_Bone_12_V02_ringfing",
           "Base_Bone_12_V02_middlefinger","Base_Bone_12_V02_pointerfinger_and_thumb_attachment"]:
    pr = stage.GetPrimAtPath(f"{PALM}/{nm}")
    c,mn,mx,n = centroid_extent(pr)
    bigs[nm]=c
    print(f"  {nm:52s} c=({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f}) ext=({mx[0]-mn[0]:.3f},{mx[1]-mn[1]:.3f},{mx[2]-mn[2]:.3f})")

print("===== SMALL bones (Palm_bone1/*) =====")
smalls={}
for nm in ["Palm_bone1","Palm_bone1_01","Palm_bone1_02","Palm_bone1_03","Palm_bone1_04"]:
    pr = stage.GetPrimAtPath(f"{PALM}/Palm_bone1/{nm}")
    if not pr or not pr.IsValid():
        print(f"  {nm}: MISSING"); continue
    c,mn,mx,n = centroid_extent(pr)
    smalls[nm]=c
    print(f"  {nm:16s} c=({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f}) ext=({mx[0]-mn[0]:.3f},{mx[1]-mn[1]:.3f},{mx[2]-mn[2]:.3f}) verts={n}")

def nearest_big(c):
    return min(bigs, key=lambda k: np.linalg.norm(bigs[k]-c))

print("===== ALL revolute joints (body0 -> body1, world anchor) =====")
for p in stage.Traverse():
    if p.GetTypeName() == "PhysicsRevoluteJoint":
        j = UsdPhysics.RevoluteJoint(p)
        b0 = j.GetBody0Rel().GetTargets(); b1 = j.GetBody1Rel().GetTargets()
        b0 = b0[0].name if b0 else "-"; b1 = b1[0].name if b1 else "-"
        axis = j.GetAxisAttr().Get()
        lp = j.GetLocalPos0Attr().Get()
        # world anchor via body0 xform
        t0 = j.GetBody0Rel().GetTargets()
        wa = None
        if t0:
            bp = stage.GetPrimAtPath(t0[0])
            M = np.array(xc.GetLocalToWorldTransform(bp), dtype=np.float64).reshape(4,4)
            lp4 = np.array([lp[0],lp[1],lp[2],1.0]) if lp else np.array([0,0,0,1.0])
            wa = (lp4 @ M)[:3]
        wstr = f"({wa[0]:+.4f},{wa[1]:+.4f},{wa[2]:+.4f})" if wa is not None else "?"
        print(f"  {p.GetName():14s} {b0:42s} -> {b1:36s} axis={axis} anchorW={wstr}")

print("===== finger ROOTS (joints whose body0 is Palm_rigid) -> nearest big bone =====")
for p in stage.Traverse():
    if p.GetTypeName() == "PhysicsRevoluteJoint":
        j = UsdPhysics.RevoluteJoint(p)
        t0 = j.GetBody0Rel().GetTargets(); t1 = j.GetBody1Rel().GetTargets()
        if t0 and t0[0].name == "Palm_rigid" and t1:
            child = stage.GetPrimAtPath(t1[0])
            ce = centroid_extent(child)
            if ce is None:
                print(f"  {p.GetName()} -> {t1[0].name}: (no mesh)"); continue
            c = ce[0]
            print(f"  {p.GetName():12s} -> {t1[0].name:34s} childC=({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f})  nearestBig={nearest_big(c)}")
