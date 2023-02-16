# -----------------------------------------------------------------------------
#
#  Gmsh Python extended tutorial 1
#
#  Geometry and mesh data
#
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# The Python API allows to do much more than what can be done in .geo
# files. These additional features are introduced gradually in the extended
# tutorials, starting with `x1.py'.

# In this first extended tutorial, we start by using the API to access basic
# geometrical and mesh data.

import gmsh
import sys

if len(sys.argv) < 2:
    print("Usage: " + sys.argv[0] + " file")
    file = 't1.msh'
    #  exit
else:
    file = sys.argv[1]

gmsh.initialize()

# You can run this tutorial on any file that Gmsh can read, e.g. a mesh file in
# the MSH format: `python t1.py file.msh'

#  gmsh.open(sys.argv[1])
gmsh.open(file)

# Print the model name and dimension:
print('Model ' + gmsh.model.getCurrent() + ' (' +
      str(gmsh.model.getDimension()) + 'D)')

# Geometrical data is made of elementary model `entities', called `points'
# (entities of dimension 0), `curves' (entities of dimension 1), `surfaces'
# (entities of dimension 2) and `volumes' (entities of dimension 3). As we have
# seen in the other Python tutorials, elementary model entities are identified
# by their dimension and by a `tag': a strictly positive identification
# number. Model entities can be either CAD entities (from the built-in `geo'
# kernel or from the OpenCASCADE `occ' kernel) or `discrete' entities (defined
# by a mesh). `Physical groups' are collections of model entities and are also
# identified by their dimension and by a tag.

# Get all the elementary entities in the model, as a vector of (dimension, tag)
# pairs:
entities = gmsh.model.getEntities()

for e in entities:
    # Dimension and tag of the entity:
    dim = e[0]
    tag = e[1]

    # Mesh data is made of `elements' (points, lines, triangles, ...), defined
    # by an ordered list of their `nodes'. Elements and nodes are identified by
    # `tags' as well (strictly positive identification numbers), and are stored
    # ("classified") in the model entity they discretize. Tags for elements and
    # nodes are globally unique (and not only per dimension, like entities).

    # A model entity of dimension 0 (a geometrical point) will contain a mesh
    # element of type point, as well as a mesh node. A model curve will contain
    # line elements as well as its interior nodes, while its boundary nodes will
    # be stored in the bounding model points. A model surface will contain
    # triangular and/or quadrangular elements and all the nodes not classified
    # on its boundary or on its embedded entities. A model volume will contain
    # tetrahedra, hexahedra, etc. and all the nodes not classified on its
    # boundary or on its embedded entities.

    # Get the mesh nodes for the entity (dim, tag):
    nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(dim, tag)

    # Get the mesh elements for the entity (dim, tag):
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, tag)

    #  if (dim,tag) == (2,1):
        #  print(len(elemTypes),elemTags[0].shape,elemNodeTags[0].shape)

    # Elements can also be obtained by type, by using `getElementTypes()'
    # followed by `getElementsByType()'.

    # Let's print a summary of the information available on the entity and its
    # mesh.

    # * Type and name of the entity:
    type = gmsh.model.getType(e[0], e[1])
    name = gmsh.model.getEntityName(e[0], e[1])
    if len(name): name += ' '
    print("Entity " + name + str(e) + " of type " + type)

    # * Number of mesh nodes and elements:
    numElem = sum(len(i) for i in elemTags)
    print(" - Mesh has " + str(len(nodeTags)) + " nodes and " + str(numElem) +
          " elements")

    # * Upward and downward adjacencies:
    up, down = gmsh.model.getAdjacencies(e[0], e[1])
    if len(up):
        print(" - Upward adjacencies: " + str(up))
    if len(down):
        print(" - Downward adjacencies: " + str(down))

    # * Does the entity belong to physical groups?
    physicalTags = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
    if len(physicalTags):
        s = ''
        for p in physicalTags:
            n = gmsh.model.getPhysicalName(dim, p)
            if n: n += ' '
            s += n + '(' + str(dim) + ', ' + str(p) + ') '
        print(" - Physical groups: " + s)

    # * Is the entity a partition entity? If so, what is its parent entity?
    partitions = gmsh.model.getPartitions(e[0], e[1])
    if len(partitions):
        print(" - Partition tags: " + str(partitions) + " - parent entity " +
              str(gmsh.model.getParent(e[0], e[1])))

    # * List all types of elements making up the mesh of the entity:
    for t in elemTypes:
        name, dim, order, numv, parv, _ = gmsh.model.mesh.getElementProperties(
            t)
        print(" - Element type: " + name + ", order " + str(order) + " (" +
              str(numv) + " nodes in param coord: " + str(parv) + ")")


    #  nodes = nodeCoords.reshape((-1,3))
    #  plt.plot(nodes[:,0],nodes[:,1],'o',mfc='w')

# We can use this to clear all the model data:
#  gmsh.clear()

#  gmsh.finalize()

#  nodeTags, nodeCoords_, nodeParams = gmsh.model.mesh.getNodes(1,4)
#  nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(2,1)
#  #  elemTags_, nodeTags_ = gmsh.model.mesh.getElementsByType(1)

#  nodes = nodeCoords.reshape((-1,3))
#  plt.plot(nodes[:,0],nodes[:,1],'ko',mfc='w')
#  nodes = nodeCoords_.reshape((-1,3))
#  plt.plot(nodes[:,0],nodes[:,1],'r*',mfc='w')



# ------------------------------------------------------------------------------

class Element:
    def __init__(self,ids):
        self.NodeIDs = ids
        self.Nodes = []
        self.Edges = []

class Edge:
    def __init__(self,ids):
        self.NodeIDs = ids
        self.Nodes = []
        self.Elements = []

# ------------------------------------------------------------------------------

NodesIds, NodesFlat, _ = gmsh.model.mesh.getNodes()
NodesXYZ = NodesFlat.reshape((-1,3))
ElemTags, ElemNodeTagsFlat = gmsh.model.mesh.getElementsByType(2)
ElemNodeTags = ElemNodeTagsFlat.reshape((-1,3)).astype(int)

cs = 'rgbcymk'

if False:
    for i,Elem in enumerate(ElemNodeTags):
        c = cs[i%len(cs)]
        Vertices = NodesXYZ[Elem-1]
        plt.plot(Vertices[:,0],Vertices[:,1],c+'o',mfc='none')
        Vertices = np.vstack((Vertices,Vertices[0,:]))
        plt.plot(Vertices[:,0],Vertices[:,1],c+'-')

    plt.show()


Elements = []
Edges = []
EdgeMap = {}

Offset = 10**int(np.log10(len(NodesXYZ))+1)

for i,Elem in enumerate(ElemNodeTags):
    Elements.append( Element(Elem-1) ) 
    Elements[-1].Nodes = NodesXYZ[Elem-1]
    for j in range(len(Elem)):
        N0 = min( Elem[j], Elem[(j+1)%len(Elem)] ) - 1
        N1 = max( Elem[j], Elem[(j+1)%len(Elem)] ) - 1
        edgeId = N1 * Offset + N0
        if edgeId not in EdgeMap:
            edge = Edge(np.array([N0, N1]))
            Edges.append(edge)
            Edges[-1].Nodes.append(NodesXYZ[N0])
            Edges[-1].Nodes.append(NodesXYZ[N1])
            EdgeMap[edgeId] = len(Edges) - 1
        else:
            edge = Edges[EdgeMap[edgeId]]
        Elements[-1].Edges.append(edge)
        Edges[EdgeMap[edgeId]].Elements.append(Elements[-1])
    
plt.xlim(-0.05,0.15)
plt.ylim(-0.02,0.32)

Centroids = []
for Elem in Elements:
    Centroids.append(np.sum(Elem.Nodes, axis=0))

Centroids = np.array(Centroids) / 3.0

Num = len(Centroids)

minx,maxx = min(NodesXYZ[:,0]),max(NodesXYZ[:,0])
miny,maxy = min(NodesXYZ[:,1]),max(NodesXYZ[:,1])

dx = maxx - minx
dy = maxy - miny
HNum = int(Num/2)
HNum = int(Num/4)
HNum = int(Num/7)
HNum = 20
#  HNum = int(Num/10)

KMeans = np.random.rand(HNum,3)
KMeans[:,0] *= dx + minx
KMeans[:,1] *= dy + miny
KMeans[:,2]  = 0.0
#  plt.plot(KMeans[:,0],KMeans[:,1],'*')

for step in range(10):
    ColourIDs = []
    # assign cluster as closest
    for Centroid in Centroids:
        Dist = KMeans - Centroid
        Dist[:,2] = (Dist[:,0]**2 + Dist[:,1]**2)**0.5
        i = np.argmin(Dist[:,2])
        ColourIDs.append(i)
    ColourIDs = np.array(ColourIDs)
    # recalculate cluster centroids
    iota = np.arange(len(ColourIDs))
    KMeans_ = np.zeros_like(KMeans)
    for i in range(len(KMeans)):
        LCent = Centroids[iota[ColourIDs == i]]
        if len(LCent) > 0:
            KMeans_[i,:] = np.sum(LCent, axis=0) / len(LCent)
    KMeans = KMeans_.copy()
    #  plt.plot(KMeans[:,0],KMeans[:,1],'*')

plt.plot(KMeans[:,0],KMeans[:,1],'*')


cs = cm.coolwarm(np.linspace(0,1,HNum))
tmp = np.linspace(0,1,HNum)
np.random.shuffle(tmp)
cs = cm.gist_rainbow(tmp)

tris = []
for i,Centroid in enumerate(Centroids):
    #  plt.plot([Centroid[0]],[Centroid[1]],'o',c=cs[ColourIDs[i]])
    tris.append(Polygon(Elements[i].Nodes[:,:2], closed=True))

#  for edge in Edges:
    #  n0 = edge.Nodes[0]
    #  n1 = edge.Nodes[1]
    #  #  plt.plot([n0[0],n1[0]],[n0[1],n1[1]],'bo-',mfc='w')
    #  plt.plot([n0[0],n1[0]],[n0[1],n1[1]],'bo',mfc='w')

ax = plt.gca()
#  fig, ax = plt.subplots()
p = PatchCollection(tris, alpha=0.7, cmap=cm.gist_rainbow)
p.set_array(ColourIDs)
ax.add_collection(p)
plt.colorbar(p, ax=ax)

plt.show()


uq,ct = np.unique(ColourIDs, return_counts=True)
plt.bar(np.arange(len(ct)),ct)
plt.show()
