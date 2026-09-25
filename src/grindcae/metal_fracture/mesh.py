"""Pre-refined 2D meshes with cohesive candidate sides and vertex-sector DOFs.

Candidate sides are split before solving; no state projection during fracture.
Unsplit incident edges keep vertex sectors compatible, including crack tips.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Interface:
    background_edge: tuple[int, int]
    elements: tuple[int, int]
    side_a: tuple[int, int]
    side_b: tuple[int, int]


@dataclass(frozen=True)
class CohesiveMesh:
    points: np.ndarray
    triangles: np.ndarray
    node_background_ids: np.ndarray
    background_triangles: np.ndarray
    interfaces: tuple[Interface, ...]
    reference_area_m2: float


def _edge_owners(triangles):
    owners = {}
    for e,t in enumerate(triangles):
        for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
            owners.setdefault(tuple(sorted((int(a),int(b)))), []).append(e)
    if any(len(v)>2 for v in owners.values()):
        raise ValueError('Non-manifold triangle edge')
    return owners


def _groups(items, links):
    parent = {int(i):int(i) for i in items}
    def root(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]
            a=parent[a]
        return a
    for a,b in links:
        ra,rb=root(a),root(b)
        parent[max(ra,rb)] = min(ra,rb)
    result = {}
    for i in sorted(parent):
        result.setdefault(root(i), []).append(i)
    return list(result.values())


def split_candidate_edges(points, triangles, candidate_edges):
    p=np.asarray(points,dtype=float)
    raw=np.asarray(triangles)
    if (p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all() or
        raw.ndim!=2 or raw.shape[1]!=3 or raw.dtype.kind not in 'iu'):
        raise ValueError('Finite 2D points and integer triangles required')
    t=raw.astype(np.int64,copy=True)
    if not len(t) or t.min()<0 or t.max()>=len(p):
        raise ValueError('Invalid triangle nodes')
    xy=p[t]
    ab,ac=xy[:,1]-xy[:,0],xy[:,2]-xy[:,0]
    areas=.5*(ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0])
    if not np.isfinite(areas).all() or np.any(areas<=0):
        raise ValueError('Triangles must have positive finite orientation')
    if len({tuple(sorted(row)) for row in t})!=len(t):
        raise ValueError('Duplicate triangles')
    owners=_edge_owners(t)
    candidates=set()
    for edge in candidate_edges:
        if len(edge)!=2 or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) for i in edge):
            raise ValueError('Integer edge identities required')
        key=tuple(sorted(map(int,edge)))
        if key in candidates or len(owners.get(key,[]))!=2:
            raise ValueError('Each candidate must be a unique interior edge')
        candidates.add(key)
    incidence={n:[] for n in range(len(p))}
    for e,tri in enumerate(t):
        for n in tri:
            incidence[int(n)].append(e)
    links={n:[] for n in incidence}
    for edge,es in owners.items():
        if len(es)==2 and edge not in candidates:
            for n in edge:
                links[n].append(tuple(es))
    new_points=[]
    background=[]
    mapping={}
    for n,es in incidence.items():
        for sector in _groups(es,links[n]):
            index=len(new_points)
            new_points.append(p[n])
            background.append(n)
            for e in sector:
                mapping[e,n]=index
    split=np.array([[mapping[e,int(n)] for n in tri] for e,tri in enumerate(t)],dtype=np.int64)
    interfaces=[]
    for edge in sorted(candidates):
        a,b=owners[edge]
        interfaces.append(Interface(edge,(a,b),tuple(mapping[a,n] for n in edge),
                                    tuple(mapping[b,n] for n in edge)))
    arrays=[np.array(new_points),split,np.array(background,dtype=np.int64),t]
    for array in arrays:
        array.setflags(write=False)
    return CohesiveMesh(*arrays,tuple(interfaces),float(areas.sum()))


def material_components(mesh, *, broken_edges):
    candidates={i.background_edge for i in mesh.interfaces}
    broken={tuple(sorted(e)) for e in broken_edges}
    if not broken.issubset(candidates):
        raise ValueError('Broken edge has no cohesive interface')
    links=[tuple(es) for edge,es in _edge_owners(mesh.background_triangles).items()
           if len(es)==2 and edge not in broken]
    return tuple(tuple(g) for g in _groups(range(len(mesh.triangles)),links))


def refined_band_mesh(*, width_m, height_m, band_depth_m, fine_size_m, coarse_size_m):
    import gmsh
    from grindcae.gmsh_runtime import initialize_gmsh
    values=(width_m,height_m,band_depth_m,fine_size_m,coarse_size_m)
    if any(isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('Mesh dimensions must be positive finite numbers')
    if band_depth_m>=height_m or fine_size_m>coarse_size_m:
        raise ValueError('Invalid process band or refinement ratio')
    if width_m*height_m/fine_size_m**2>50000:
        raise ValueError('Experimental mesh size guard exceeded')
    if gmsh.isInitialized():
        raise RuntimeError('Gmsh already in use; do not mutate another active model')
    initialize_gmsh(gmsh,['grindcae-metal-band','-v','0'])
    try:
        gmsh.model.add('metal_band')
        gmsh.model.occ.addRectangle(0,0,0,width_m,height_m)
        gmsh.model.occ.synchronize()
        box=gmsh.model.mesh.field.add('Box')
        for name,val in {'VIn':fine_size_m,'VOut':coarse_size_m,'XMin':0,'XMax':width_m,
                         'YMin':height_m-band_depth_m,'YMax':height_m,'ZMin':-1,'ZMax':1,
                         'Thickness':band_depth_m}.items():
            gmsh.model.mesh.field.setNumber(box,name,val)
        gmsh.model.mesh.field.setAsBackgroundMesh(box)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints',0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',0)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0)
        gmsh.option.setNumber('Mesh.ElementOrder',1)
        gmsh.option.setNumber('Mesh.RandomSeed',1)
        gmsh.model.mesh.generate(2)
        tags,coords,_=gmsh.model.mesh.getNodes()
        order=np.argsort(tags)
        p=np.asarray(coords).reshape(-1,3)[order,:2]
        tag_map={int(tag):i for i,tag in enumerate(np.asarray(tags)[order])}
        _,node_tags=gmsh.model.mesh.getElementsByType(2)
        t=np.array([tag_map[int(tag)] for tag in node_tags],dtype=np.int64).reshape(-1,3)
        owners=_edge_owners(t)
        edges=[edge for edge,es in owners.items() if len(es)==2 and
               min(p[list(edge),1])>=height_m-band_depth_m-1e-14]
        return split_candidate_edges(p,t,edges)
    finally:
        gmsh.finalize()
