/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_BLINK_H_
#define NCCL_BLINK_H_

#include "graph.h"
#include "topo.h"

// Main entry point: compute Blink spanning tree packing for tree channels.
// Replaces ncclTopoCompute for tree graphs when NCCL_BLINK=1.
//   comm:   NCCL communicator (used for bootstrap when
//           NCCL_BLINK_GLOBAL=1; pass NULL to disable cross-host
//           topology aggregation)
//   system: NCCL topology system (read-only)
//   graph:  output ncclTopoGraph (will be populated with tree channels)
//   nChannelsNeeded: number of channels to produce (typically ringGraph->nChannels)
struct ncclComm;
ncclResult_t ncclBlinkCompute(struct ncclComm* comm,
                              struct ncclTopoSystem* system,
                              struct ncclTopoGraph* graph,
                              int nChannelsNeeded);

/*------------------------------------------------------------------------
 * Internal data structures
 *------------------------------------------------------------------------*/

#define BLINK_MAX_GPUS 16
#define BLINK_MAX_EDGES (BLINK_MAX_GPUS * BLINK_MAX_GPUS)
#define BLINK_MAX_TREES MAXCHANNELS

struct BlinkEdge {
  int src;
  int dst;
  float capacity;   // link bandwidth in GB/s
};

// Logical GPU-only graph consumed by MWU + Edmonds. Edges (i,j) in this
// graph correspond to bottleneck-paths through the full physical topology
// once the M2 path solver runs; on a pure-NVL topology they are direct
// LINK_NVL edges as before.
struct BlinkGraph {
  int nVertices;
  int nEdges;
  BlinkEdge edges[BLINK_MAX_EDGES];
  int gpuRanks[BLINK_MAX_GPUS];  // vertex index -> NCCL rank
  int edgeIndex[BLINK_MAX_GPUS][BLINK_MAX_GPUS]; // edgeIndex[src][dst] -> edge index, -1 if none
};

struct BlinkTree {
  int root;
  int parent[BLINK_MAX_GPUS];    // parent[v] for arborescence, parent[root] = -1
  float weight;                   // assigned rate for this tree
};

struct BlinkPacking {
  int nTrees;
  BlinkTree trees[BLINK_MAX_TREES];
  float totalRate;
};

/*------------------------------------------------------------------------
 * Full physical topology graph (M1). Holds every GPU/NVS/PCI/CPU/NIC/NET
 * vertex within scope and every directed link between them, plus a
 * synthetic FABRIC vertex (added in M4) for the inter-host black-box
 * fabric. The MWU/Edmonds inner loop never sees this graph directly --
 * it sees the GPU-only `BlinkGraph` above, whose edges carry the
 * bottleneck capacity of the corresponding shortest path through the
 * physical graph (computed in M2). Refinement, however, tracks residual
 * capacities on physical edges so that shared switches/NICs are not
 * over-packed.
 *------------------------------------------------------------------------*/

#define BLINK_MAX_PHYS_VERTICES 128
#define BLINK_MAX_PHYS_EDGES    1024
#define BLINK_MAX_PATH_HOPS     16

#define BLINK_VTX_GPU     0
#define BLINK_VTX_NVS     1
#define BLINK_VTX_PCI     2
#define BLINK_VTX_CPU     3
#define BLINK_VTX_NIC     4
#define BLINK_VTX_NET     5
#define BLINK_VTX_FABRIC  6

// Synthetic link kind for NET<->FABRIC edges introduced in M4. Lives in
// its own value space so it does not collide with NCCL's LINK_* enum.
#define BLINK_LINK_FAB    100

struct BlinkPhysVertex {
  int kind;          // BLINK_VTX_*
  int hostId;        // NCCL_TOPO_ID_SYSTEM_ID(node->id), -1 for FABRIC
  int64_t topoId;    // ncclTopoNode->id, or sentinel for FABRIC
  int gpuLocalIdx;   // when kind==GPU, local-host index into gpuRanks[]; else -1
  int rank;          // when kind==GPU, NCCL global rank; else -1
  float bw;          // when kind==NET, NET line rate in GB/s; else 0
};

struct BlinkPhysEdge {
  int src;           // vertex index
  int dst;           // vertex index
  float capacity;    // GB/s
  int kind;          // LINK_NVL, LINK_PCI, LINK_NET, LINK_SYS, BLINK_LINK_FAB, ...
};

struct BlinkPhysGraph {
  int nVertices;
  int nEdges;
  int nGpus;
  int gpuVtxIdx[BLINK_MAX_GPUS];   // gpuVtxIdx[i] = vertex index for GPU local i
  int gpuRanks[BLINK_MAX_GPUS];    // gpuRanks[i] = NCCL rank for GPU local i
  struct BlinkPhysVertex vertices[BLINK_MAX_PHYS_VERTICES];
  struct BlinkPhysEdge edges[BLINK_MAX_PHYS_EDGES];
};

struct BlinkPath {
  int nHops;
  int edges[BLINK_MAX_PATH_HOPS];   // physical edge indices, src->dst order
  float bottleneck;                  // min capacity along the path, GB/s
};

struct BlinkPathTable {
  // paths[i][j] = path from GPU local i to GPU local j; i==j -> nHops=0
  struct BlinkPath paths[BLINK_MAX_GPUS][BLINK_MAX_GPUS];
};

#endif // NCCL_BLINK_H_
