# Full dump of every revolute joint's frames/limits/drive + finger body masses,
# so the finger chains can be transcribed faithfully into Newton.
import os, numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

# Assets ship in ../assets of the DexiGrab repo; override with DEXIGRAB_ASSETS.
_ASSETS = os.environ.get("DEXIGRAB_ASSETS") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "assets")
U = os.path.join(_ASSETS, "robot_hand_flexpalm.usda")
stage = Usd.Stage.Open(U)
xc = UsdGeom.XformCache()

def q_str(q):
    if q is None: return "None"
    return f"({q.GetReal():+.3f},{q.GetImaginary()[0]:+.3f},{q.GetImaginary()[1]:+.3f},{q.GetImaginary()[2]:+.3f})"

print("===== Palm_rigid world transform =====")
pr = stage.GetPrimAtPath("/Robotic_Hand_V5_simulacra/Palm_rigid")
M = xc.GetLocalToWorldTransform(pr)
print("  translate:", [round(x,5) for x in M.ExtractTranslation()])
print("  rot quat :", q_str(M.ExtractRotationQuat()))

print("\n===== every PhysicsRevoluteJoint: frames / limits / drive =====")
for p in stage.Traverse():
    if p.GetTypeName() != "PhysicsRevoluteJoint":
        continue
    j = UsdPhysics.RevoluteJoint(p)
    b0 = j.GetBody0Rel().GetTargets(); b1 = j.GetBody1Rel().GetTargets()
    b0 = b0[0].name if b0 else "-"; b1 = b1[0].name if b1 else "-"
    axis = j.GetAxisAttr().Get()
    lp0 = j.GetLocalPos0Attr().Get(); lp1 = j.GetLocalPos1Attr().Get()
    lr0 = j.GetLocalRot0Attr().Get(); lr1 = j.GetLocalRot1Attr().Get()
    lo = j.GetLowerLimitAttr().Get(); hi = j.GetUpperLimitAttr().Get()
    # drive (angular)
    drive = UsdPhysics.DriveAPI.Get(p, "angular")
    dt = ds = dd = dmf = None
    if drive:
        dt = drive.GetTargetPositionAttr().Get()
        ds = drive.GetStiffnessAttr().Get()
        dd = drive.GetDampingAttr().Get()
        dmf = drive.GetMaxForceAttr().Get()
    print(f"\n  {p.GetName()}  {b0} -> {b1}  axis={axis}  limit=[{lo},{hi}]")
    print(f"    localPos0={[round(x,5) for x in lp0] if lp0 else None}  localRot0={q_str(lr0)}")
    print(f"    localPos1={[round(x,5) for x in lp1] if lp1 else None}  localRot1={q_str(lr1)}")
    print(f"    drive: target={dt} stiffness={ds} damping={dd} maxForce={dmf}")

print("\n===== finger body masses (UsdPhysics.MassAPI) =====")
finger_bodies = ["Group_1","second_thumb_hinge","third_thumb_hinge","Metacarpal_Bone_V02_03",
    "Proximal_Phalanx_Bone_V02_01","Distal_Phalanx_Bone_V02_01",
    "Base_Bone_1_V02_01","Metacarpal_Bone_V02_pinky","Proximal_Phalanx_Bone_V02_pinky","Distal_Phalanx_Bone_V02_02",
    "Base_Bone_1_V02_02","Metacarpal_Bone_V02_02","Proximal_Phalanx_Bone_V02","Distal_Phalanx_Bone_V02",
    "Base_Bone_1_V02_03","Metacarpal_Bone_V02","Proximal_Phalanx_Bone_V02_middlefinger","Distal_Phalanx_Bone_V02_03",
    "Base_Bone_1_V02","Metacarpal_Bone_V02_01","Proximal_Phalanx_Bone_V02_02","Distal_Phalanx_Bone_V02_04"]
for nm in finger_bodies:
    path = f"/Robotic_Hand_V5_simulacra/{nm}"
    pp = stage.GetPrimAtPath(path)
    if not pp or not pp.IsValid():
        print(f"  {nm}: MISSING"); continue
    mapi = UsdPhysics.MassAPI.Get(stage, path)
    mass = mapi.GetMassAttr().Get() if mapi else None
    has_rb = pp.HasAPI(UsdPhysics.RigidBodyAPI)
    # world transform
    Mb = xc.GetLocalToWorldTransform(pp)
    t = Mb.ExtractTranslation()
    print(f"  {nm:40s} mass={mass} rigidBody={has_rb} worldT=({t[0]:+.4f},{t[1]:+.4f},{t[2]:+.4f})")
