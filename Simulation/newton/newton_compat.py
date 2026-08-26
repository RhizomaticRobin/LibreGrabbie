"""Newton 1.2 -> 1.4 compatibility for this codebase.

This sim line was written against Newton 1.2.0, whose ``ModelBuilder`` exposed
``add_equality_constraint_connect`` / ``add_equality_constraint_weld``. Newton 1.4
moved equality constraints into the ``mujoco:equality_constraint`` custom-attribute
namespace (``newton._src.solvers.mujoco.equality``) and dropped the builder methods.
Importing this module re-attaches the two 1.2-era methods, forwarding to the 1.4
helper 1:1 (``mujoco:eq_solref`` custom attributes pass through unchanged — 1.4
still registers that exact name). On Newton 1.2 this module is a no-op.
"""
from newton import ModelBuilder

if not hasattr(ModelBuilder, "add_equality_constraint_connect"):
    from newton._src.solvers.mujoco.enums import EqType
    from newton._src.solvers.mujoco.equality import _add_equality_constraint

    def _connect(self, body1=-1, body2=-1, anchor=None, label=None, enabled=True,
                 custom_attributes=None):
        return _add_equality_constraint(self, EqType.CONNECT, body1=body1, body2=body2,
                                        anchor=anchor, label=label, enabled=enabled,
                                        custom_attributes=custom_attributes)

    def _weld(self, body1=-1, body2=-1, anchor=None, torquescale=None, relpose=None,
              label=None, enabled=True, custom_attributes=None):
        return _add_equality_constraint(self, EqType.WELD, body1=body1, body2=body2,
                                        anchor=anchor, torquescale=torquescale,
                                        relpose=relpose, label=label, enabled=enabled,
                                        custom_attributes=custom_attributes)

    ModelBuilder.add_equality_constraint_connect = _connect
    ModelBuilder.add_equality_constraint_weld = _weld

from newton import Model

if not hasattr(Model, "equality_constraint_count"):
    # 1.2's Model.equality_constraint_count; in 1.4 the rows live under the
    # model's ``mujoco`` custom-attribute namespace.
    def _eq_count(self):
        ns = getattr(self, "mujoco", None)
        arr = getattr(ns, "equality_constraint_type", None) if ns is not None else None
        return 0 if arr is None else len(arr)

    Model.equality_constraint_count = property(_eq_count)
