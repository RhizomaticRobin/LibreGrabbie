# Enumerate every scene-graph Mesh under the hand root, its owning body Xform,
# whether it has RigidBodyAPI, world centroid + z-range, and whether it's already
# loaded by flexpalm_bones4.py. The leftovers are "the rest of the parts".
import os, numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Sdf

# Assets ship in ../assets of the DexiGrab repo; override with DEXIGRAB_ASSETS.
_ASSETS = os.environ.get("DEXIGRAB_ASSETS") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "assets")
U = os.path.join(_ASSETS, "robot_hand_flexpalm.usda")
ROOT = "/Robotic_Hand_V5_simulacra"
stage = Usd.Stage.Open(U)
xc = UsdGeom.XformCache()

LOADED_BIG = {"Base_Bone_12_V02_pinky","Base_Bone_12_V02_ringfing","Base_Bone_12_V02_middlefinger",
              "Base_Bone_12_V02_pointerfinger_and_thumb_attachment"}
LOADED_SMALL = {"Palm_bone1","Palm_bone1_01","Palm_bone1_03","Palm_bone1_04"}
LOADED_FINGERS = {"Group_1","second_thumb_hinge","third_thumb_hinge","Metacarpal_Bone_V02_03",
    "Proximal_Phalanx_Bone_V02_01","Distal_Phalanx_Bone_V02_01","Base_Bone_1_V02_01",
    "Metacarpal_Bone_V02_pinky","Proximal_Phalanx_Bone_V02_pinky","Distal_Phalanx_Bone_V02_02",
    "Base_Bone_1_V02_02","Metacarpal_Bone_V02_02","Proximal_Phalanx_Bone_V02","Distal_Phalanx_Bone_V02",
    "Base_Bone_1_V02_03","Metacarpal_Bone_V02","Proximal_Phalanx_Bone_V02_middlefinger",
    "Distal_Phalanx_Bone_V02_03","Base_Bone_1_V02","Metacarpal_Bone_V02_01",
    "Proximal_Phalanx_Bone_V02_02","Distal_Phalanx_Bone_V02_04"}

def centroid(prim):
    pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
    M = np.array(xc.GetLocalToWorldTransform(prim), dtype=np.float64).reshape(4,4)
    w = (np.concatenate([pts, np.ones((len(pts),1))],1) @ M)[:,:3]
    return w.mean(0), w.min(0), w.max(0), len(pts)

def owning_body(prim):
    """nearest ancestor (or self) carrying RigidBodyAPI, else the top Xform under ROOT."""
    p = prim
    while p and str(p.GetPath()) != ROOT:
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p.GetName(), True
        p = p.GetParent()
    # top-level segment under ROOT
    rel = str(prim.GetPath())[len(ROOT)+1:]
    return rel.split("/")[0], False

print(f"{'mesh path (rel)':70s} {'ownerBody':32s} {'rb':3s} {'verts':>6s}  centroidZ  status")
seen_owners = {}
for p in stage.Traverse():
    if p.GetTypeName() != "Mesh": continue
    path = str(p.GetPath())
    if not path.startswith(ROOT + "/"): continue
    # skip the prototype library 'over' blocks (not in active scene composition)
    if p.GetSpecifier() == Sdf.SpecifierOver: continue
    c,mn,mx,n = centroid(p)
    owner, has_rb = owning_body(p)
    if owner in LOADED_BIG: status="LOADED big"
    elif owner in LOADED_SMALL: status="LOADED small"
    elif owner in LOADED_FINGERS: status="LOADED finger"
    elif owner=="Palm_bone1_02": status="skip degenerate"
    else: status="*** REMAINING ***"
    rel = path[len(ROOT)+1:]
    print(f"{rel:70s} {owner:32s} {'Y' if has_rb else '-':3s} {n:6d}  z[{mn[2]:+.3f},{mx[2]:+.3f}] {status}")
    seen_owners.setdefault(owner, status)

print("\n===== owners summary =====")
for o,s in sorted(seen_owners.items(), key=lambda kv: kv[1]):
    print(f"  {o:36s} {s}")
print("\n===== global z-range of ALL scene meshes (for lift) =====")
allz=[]
for p in stage.Traverse():
    if p.GetTypeName()=="Mesh" and str(p.GetPath()).startswith(ROOT+"/") and p.GetSpecifier()!=Sdf.SpecifierOver:
        pts=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(),dtype=np.float64)
        M=np.array(xc.GetLocalToWorldTransform(p),dtype=np.float64).reshape(4,4)
        w=(np.concatenate([pts,np.ones((len(pts),1))],1)@M)[:,:3]
        allz.append((w[:,2].min(), w[:,2].max()))
allz=np.array(allz)
print(f"  min z = {allz[:,0].min():+.4f}   max z = {allz[:,1].max():+.4f}")
