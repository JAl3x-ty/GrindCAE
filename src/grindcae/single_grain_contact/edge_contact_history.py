"""Background-edge endpoint ownership, independent of legacy v1/v2 node keys.

This module does not prescribe an activation law or migrate ambiguous nodal history.
Tractions supplied to the integrator are global vectors, invariant under edge reversal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from .contact_law import ContactPointState


@dataclass(frozen=True)
class CircleEdgeIntegral:
    force_N: np.ndarray
    residual_tangent_N_per_m: np.ndarray
    energy_J: float
    active_interval: tuple[float, float] | None
    maximum_penetration_m: float = 0.


def integrate_circle_penalty_edge(coordinates, displacement, *, center, radius,
                                  penalty, thickness, order):
    """Integrate a frictionless full-circle penalty potential on a P1 edge.

    Reference length is fixed. Exact circle/chord intersections split the
    active domain; the moving-domain terms vanish because pressure and potential
    are zero at its boundary. No persistent quadrature history is implied.
    """
    x=np.asarray(coordinates,dtype=float)
    u=np.asarray(displacement,dtype=float)
    center=np.asarray(center,dtype=float)
    if x.shape!=(2,2) or u.shape!=(2,2) or center.shape!=(2,):
        raise ValueError('invalid circle edge coordinate shape')
    if not all(np.all(np.isfinite(v)) for v in (x,u,center)):
        raise ValueError('nonfinite circle edge coordinates')
    if any(not math.isfinite(v) or v<=0 for v in (radius,penalty,thickness)):
        raise ValueError('circle edge parameters must be positive finite')
    if type(order) is not int or not 1<=order<=32:
        raise ValueError('invalid circle edge quadrature order')
    length=float(np.linalg.norm(x[1]-x[0]))
    offset=x[0]+u[0]-center
    direction=x[1]+u[1]-x[0]-u[0]
    norm2=float(direction@direction)
    if length<=0 or norm2<=0:
        raise ValueError('degenerate circle contact edge')
    force=np.zeros(4); tangent=np.zeros((4,4)); energy=0.
    middle=-float(offset@direction)/norm2
    closest=offset+middle*direction
    half_squared=(radius**2-float(closest@closest))/norm2
    if half_squared<=0:
        return CircleEdgeIntegral(force,tangent,energy,None)
    half=math.sqrt(half_squared)
    lo,hi=max(0.,middle-half),min(1.,middle+half)
    if hi<=lo:
        return CircleEdgeIntegral(force,tangent,energy,None)
    closest_fraction=min(hi,max(lo,middle))
    if np.linalg.norm(offset+closest_fraction*direction)<=64*np.finfo(float).eps*radius:
        raise ValueError('circle contact edge crosses nonunique center')
    nodes,weights=np.polynomial.legendre.leggauss(order)
    for node,weight in zip(nodes,weights,strict=True):
        fraction=lo+.5*(node+1)*(hi-lo)
        radial=offset+fraction*direction
        distance=float(np.linalg.norm(radial)); normal=radial/distance
        penetration=radius-distance
        shape=np.hstack(((1-fraction)*np.eye(2),fraction*np.eye(2)))
        measure=.5*(hi-lo)*weight*length*thickness
        pressure=penalty*penetration
        local=penalty*np.outer(normal,normal)-pressure/distance*(np.eye(2)-np.outer(normal,normal))
        force+=measure*shape.T@(pressure*normal)
        tangent+=measure*shape.T@local@shape
        energy+=measure*.5*penalty*penetration**2
    maximum_penetration=radius-float(np.linalg.norm(offset+closest_fraction*direction))
    return CircleEdgeIntegral(force,tangent,float(energy),(lo,hi),maximum_penetration)


@dataclass(frozen=True)
class EdgeGaussPoint:
    key: tuple[int,int,int,int,int,int]
    reference_point_m: tuple[float,float]
    shape_weights: tuple[float,float]
    weight_m: float


def edge_gauss_points(edges,coordinates,background_nodes,*,order,panels):
    """Fixed reference-edge integration; history identity includes the rule.

    Changing this rule during a trajectory requires an explicit conservative
    history transfer and is not performed by this function.
    """
    if type(order) is not int or not 1<=order<=32 or type(panels) is not int or panels<1:
        raise ValueError('invalid edge quadrature order or panel count')
    coordinates=np.asarray(coordinates,dtype=float)
    if coordinates.shape!=(len(background_nodes),2) or not np.all(np.isfinite(coordinates)) or len(set(background_nodes))!=len(background_nodes):
        raise ValueError('invalid background edge coordinates')
    lookup=dict(zip(background_nodes,coordinates,strict=True))
    canonical=sorted({key[:2] for key in endpoint_keys(edges)})
    nodes,weights=np.polynomial.legendre.leggauss(order)
    result=[]
    for a,b in canonical:
        if a not in lookup or b not in lookup: raise ValueError('unknown background edge node')
        first,last=lookup[a],lookup[b]
        length=float(np.linalg.norm(last-first))
        if length<=0: raise ValueError('zero length integration edge')
        for panel in range(panels):
            for index,(node,weight) in enumerate(zip(nodes,weights,strict=True)):
                fraction=(panel+.5*(node+1.))/panels
                point=(1.-fraction)*first+fraction*last
                result.append(EdgeGaussPoint((a,b,order,panels,panel,index),
                    tuple(float(v) for v in point),(1.-fraction,fraction),.5*length*weight/panels))
    return tuple(result)


def endpoint_keys(edges):
    canonical = []
    for a, b in edges:
        if isinstance(a, bool) or isinstance(b, bool) or int(a) != a or int(b) != b:
            raise ValueError('background node identities must be integers')
        a, b = sorted((int(a), int(b)))
        if a < 0 or a == b:
            raise ValueError('edge requires two distinct nonnegative background nodes')
        canonical.append((a, b))
    if len(set(canonical)) != len(canonical):
        raise ValueError('duplicate background edge')
    return tuple(key for a,b in sorted(canonical) for key in ((a,b,a),(a,b,b)))


def map_endpoint_history(previous, new_edges):
    """Return private active/archived histories; new edges start independently."""
    keys = endpoint_keys(new_edges)
    for key, value in previous.items():
        if (len(key) != 3 or key[0] >= key[1] or key[2] not in key[:2]
                or not isinstance(value, ContactPointState)):
            raise ValueError('invalid endpoint history')
    active = {key: previous.get(key, ContactPointState()) for key in keys}
    archived = {key: value for key,value in previous.items() if key not in active}
    return active, archived


def integrate_endpoints(edges, lengths, global_tractions, energy_densities, *, thickness):
    """Equivalent two-endpoint boundary integral; multiply thickness exactly once."""
    edges = tuple(edges)
    keys = endpoint_keys(edges)
    if set(keys) != set(global_tractions) or set(keys) != set(energy_densities):
        raise ValueError('endpoint data do not match boundary')
    if not math.isfinite(thickness) or thickness <= 0:
        raise ValueError('thickness must be positive')
    force, energies = {}, []
    for (a,b), length in zip(edges,lengths,strict=True):
        if not math.isfinite(length) or length <= 0:
            raise ValueError('edge length must be positive')
        lo, hi = sorted((int(a),int(b)))
        for node in (lo,hi):
            key = (lo,hi,node)
            traction = np.asarray(global_tractions[key],dtype=float)
            density = float(energy_densities[key])
            if traction.shape != (2,) or not np.all(np.isfinite(traction)) or not math.isfinite(density) or density < 0:
                raise ValueError('invalid endpoint contribution')
            weight = .5*length*thickness
            force[node] = force.get(node,np.zeros(2)) + weight*traction
            energies.append(weight*density)
    return force, math.fsum(energies)


def audit_pure_mapping(old_force, new_force, old_energy, new_energy):
    """Check the entire background vector; equal resultant alone is insufficient."""
    keys = sorted(set(old_force) | set(new_force))
    old = np.array([old_force.get(k,np.zeros(2)) for k in keys],dtype=float)
    new = np.array([new_force.get(k,np.zeros(2)) for k in keys],dtype=float)
    if not np.all(np.isfinite(old)) or not np.all(np.isfinite(new)):
        raise ValueError('nonfinite force')
    tolerance = 64*np.finfo(float).eps*max(float(np.sum(np.abs(old))),float(np.sum(np.abs(new))))
    difference = float(np.linalg.norm(new-old))
    if difference > tolerance:
        raise ValueError('pure mapping changes background force and virtual work')
    if not all(math.isfinite(v) and v >= 0 for v in (old_energy,new_energy)):
        raise ValueError('invalid energy')
    energy_jump = new_energy-old_energy
    if abs(energy_jump) > 64*np.finfo(float).eps*(abs(old_energy)+abs(new_energy)):
        raise ValueError('pure mapping changes stored energy')
    return dict(force_difference_N=difference,energy_jump_J=energy_jump)
