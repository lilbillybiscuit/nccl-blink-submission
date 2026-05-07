/*************************************************************************
 * Alex/Micah/Bill
 *************************************************************************/

#include "blink.h"
#include "edmonds.h"
#include "bootstrap.h"
#include "comm.h"
#include "core.h"
#include "graph.h"
#include "topo.h"
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

// Forward declaration: implemented in src/graph/search.cc (no header).
ncclResult_t ncclTopoSelectNets(struct ncclTopoSystem* system,
                                int typeInter, int gpu,
                                int nets[NCCL_TOPO_MAX_NODES],
                                int* netCountRet);

// Return elapsed microseconds between two timespecs
static inline double blinkElapsedUs(struct timespec* start, struct timespec* end) {
  return (double)(end->tv_sec - start->tv_sec) * 1e6 +
         (double)(end->tv_nsec - start->tv_nsec) / 1e3;
}

/*========================================================================
 * Section 1: Graph Extraction
 *========================================================================*/

// Find the index of a GPU node by pointer within the GPU node set
static int blinkGpuIndex(struct ncclTopoSystem* system, struct ncclTopoNode* node) {
  return (int)(node - system->nodes[GPU].nodes);
}

// Extract directed GPU interconnect graph from ncclTopoSystem.
// Uses NCCL's computed paths (which resolve NVSwitch hops) rather than
// raw links, so this works on both direct NVLink and NVSwitch topologies.
static ncclResult_t blinkExtractGraph(struct ncclTopoSystem* system,
                                      struct BlinkGraph* bg) {
  memset(bg, 0, sizeof(struct BlinkGraph));
  memset(bg->edgeIndex, -1, sizeof(bg->edgeIndex));
  int nGpus = system->nodes[GPU].count;
  if (nGpus > BLINK_MAX_GPUS) {
    WARN("Blink: too many GPUs (%d > %d)", nGpus, BLINK_MAX_GPUS);
    return ncclInternalError;
  }
  bg->nVertices = nGpus;

  for (int i = 0; i < nGpus; i++) {
    bg->gpuRanks[i] = system->nodes[GPU].nodes[i].gpu.rank;
  }

  // First try: walk raw links for direct GPU-to-GPU NVLinks
  for (int i = 0; i < nGpus; i++) {
    struct ncclTopoNode* gpu = &system->nodes[GPU].nodes[i];
    for (int l = 0; l < gpu->nlinks; l++) {
      struct ncclTopoLink* link = &gpu->links[l];
      if (link->type != LINK_NVL) continue;
      if (link->remNode->type != GPU) continue;
      int j = blinkGpuIndex(system, link->remNode);
      if (j < 0 || j >= nGpus) continue;

      int found = bg->edgeIndex[i][j];
      if (found >= 0) {
        bg->edges[found].capacity += link->bw;
      } else {
        if (bg->nEdges >= BLINK_MAX_EDGES) {
          WARN("Blink: too many edges");
          return ncclInternalError;
        }
        bg->edgeIndex[i][j] = bg->nEdges;
        bg->edges[bg->nEdges].src = i;
        bg->edges[bg->nEdges].dst = j;
        bg->edges[bg->nEdges].capacity = link->bw;
        bg->nEdges++;
      }
    }
  }

  // Fallback: if no direct GPU-to-GPU edges found (NVSwitch topology),
  // use NCCL's computed GPU-to-GPU paths instead.
  if (bg->nEdges == 0) {
    INFO(NCCL_GRAPH, "Blink: no direct GPU-GPU NVLinks, using computed paths (NVSwitch?)");
    for (int i = 0; i < nGpus; i++) {
      struct ncclTopoLinkList* paths = system->nodes[GPU].nodes[i].paths[GPU];
      for (int j = 0; j < nGpus; j++) {
        if (i == j) continue;
        if (paths[j].type != PATH_NVL) continue;
        if (paths[j].bw <= 0) continue;

        if (bg->nEdges >= BLINK_MAX_EDGES) {
          WARN("Blink: too many edges");
          return ncclInternalError;
        }
        bg->edgeIndex[i][j] = bg->nEdges;
        bg->edges[bg->nEdges].src = i;
        bg->edges[bg->nEdges].dst = j;
        bg->edges[bg->nEdges].capacity = paths[j].bw;
        bg->nEdges++;
      }
    }
  }

  INFO(NCCL_GRAPH, "Blink: extracted graph with %d GPUs, %d directed edges",
       bg->nVertices, bg->nEdges);

  return ncclSuccess;
}

/*------------------------------------------------------------------------
 * Full physical topology graph extraction (M1).
 *
 * Walks every GPU/NVS/PCI/CPU/NIC/NET vertex and every directed link
 * between them, building a `BlinkPhysGraph`. Self-loops (LINK_LOC) are
 * skipped, as are GIN nodes (GPU-internal networking, not modelled).
 * Capacity comes straight from `link->bw`; NIC<->NET edges already carry
 * the per-NIC line rate set in topo.cc:390-391.
 *
 * NOTE: NIC<->FABRIC (or NET<->FABRIC) edges are NOT added here. They are
 * synthesized in M4 once a single FABRIC vertex is introduced for the
 * inter-host black-box fabric.
 *------------------------------------------------------------------------*/

static int blinkVtxKindFromType(int nodeType) {
  switch (nodeType) {
    case GPU: return BLINK_VTX_GPU;
    case NVS: return BLINK_VTX_NVS;
    case PCI: return BLINK_VTX_PCI;
    case CPU: return BLINK_VTX_CPU;
    case NIC: return BLINK_VTX_NIC;
    case NET: return BLINK_VTX_NET;
    default:  return -1;   // GIN and any unknown types are skipped
  }
}

static int blinkPhysAddVertex(struct BlinkPhysGraph* pg, int kind, int hostId,
                              int64_t topoId, int gpuLocalIdx) {
  if (pg->nVertices >= BLINK_MAX_PHYS_VERTICES) return -1;
  int idx = pg->nVertices++;
  pg->vertices[idx].kind = kind;
  pg->vertices[idx].hostId = hostId;
  pg->vertices[idx].topoId = topoId;
  pg->vertices[idx].gpuLocalIdx = gpuLocalIdx;
  pg->vertices[idx].rank = -1;
  pg->vertices[idx].bw = 0.0f;
  return idx;
}

static int blinkPhysAddEdge(struct BlinkPhysGraph* pg, int src, int dst,
                            float cap, int kind) {
  if (pg->nEdges >= BLINK_MAX_PHYS_EDGES) return -1;
  int idx = pg->nEdges++;
  pg->edges[idx].src = src;
  pg->edges[idx].dst = dst;
  pg->edges[idx].capacity = cap;
  pg->edges[idx].kind = kind;
  return idx;
}

// Linear scan: small graphs (<128 vertices), called O(E) times during
// extraction. If this becomes hot in M4, swap for a (kind,topoId)->idx hash.
static int blinkPhysFindVertex(const struct BlinkPhysGraph* pg, int kind,
                               int64_t topoId) {
  for (int v = 0; v < pg->nVertices; v++) {
    if (pg->vertices[v].kind == kind && pg->vertices[v].topoId == topoId) return v;
  }
  return -1;
}

static const char* blinkVtxKindStr(int kind) {
  switch (kind) {
    case BLINK_VTX_GPU:    return "GPU";
    case BLINK_VTX_NVS:    return "NVS";
    case BLINK_VTX_PCI:    return "PCI";
    case BLINK_VTX_CPU:    return "CPU";
    case BLINK_VTX_NIC:    return "NIC";
    case BLINK_VTX_NET:    return "NET";
    case BLINK_VTX_FABRIC: return "FAB";
    default:               return "??";
  }
}

static ncclResult_t blinkExtractFullGraph(struct ncclTopoSystem* system,
                                          struct BlinkPhysGraph* pg) {
  memset(pg, 0, sizeof(struct BlinkPhysGraph));

  // Pass 1: enumerate vertices in a fixed type order so subsequent edge
  // walks find them via blinkPhysFindVertex.
  const int kindTypes[] = { GPU, NVS, PCI, CPU, NIC, NET };
  const int nKindTypes = (int)(sizeof(kindTypes) / sizeof(kindTypes[0]));
  int nGpus = 0;
  for (int kt = 0; kt < nKindTypes; kt++) {
    int t = kindTypes[kt];
    int kind = blinkVtxKindFromType(t);
    for (int n = 0; n < system->nodes[t].count; n++) {
      struct ncclTopoNode* node = &system->nodes[t].nodes[n];
      int hostId = (int)NCCL_TOPO_ID_SYSTEM_ID(node->id);
      int gpuLocalIdx = -1;
      if (t == GPU) {
        if (nGpus >= BLINK_MAX_GPUS) {
          WARN("Blink: too many GPUs (%d > %d)",
               system->nodes[GPU].count, BLINK_MAX_GPUS);
          return ncclInternalError;
        }
        gpuLocalIdx = nGpus;
      }
      int idx = blinkPhysAddVertex(pg, kind, hostId, node->id, gpuLocalIdx);
      if (idx < 0) {
        WARN("Blink: BLINK_MAX_PHYS_VERTICES (%d) exceeded",
             BLINK_MAX_PHYS_VERTICES);
        return ncclInternalError;
      }
      if (t == GPU) {
        pg->gpuVtxIdx[gpuLocalIdx] = idx;
        pg->gpuRanks[gpuLocalIdx] = node->gpu.rank;
        pg->vertices[idx].rank = node->gpu.rank;
        nGpus++;
      } else if (t == NET) {
        pg->vertices[idx].bw = node->net.bw;
      }
    }
  }
  pg->nGpus = nGpus;

  // Pass 2: walk every link out of every node and add a directed edge.
  // Skip LINK_LOC self-loops, zero-bandwidth links, and links whose remote
  // end-point is a node type we did not enumerate (e.g. GIN).
  for (int v = 0; v < pg->nVertices; v++) {
    int kind = pg->vertices[v].kind;
    int t;
    switch (kind) {
      case BLINK_VTX_GPU: t = GPU; break;
      case BLINK_VTX_NVS: t = NVS; break;
      case BLINK_VTX_PCI: t = PCI; break;
      case BLINK_VTX_CPU: t = CPU; break;
      case BLINK_VTX_NIC: t = NIC; break;
      case BLINK_VTX_NET: t = NET; break;
      default: continue;   // FABRIC is added in M4
    }

    // Resolve the original ncclTopoNode by id within its type bucket.
    struct ncclTopoNode* src = NULL;
    for (int n = 0; n < system->nodes[t].count; n++) {
      if (system->nodes[t].nodes[n].id == pg->vertices[v].topoId) {
        src = &system->nodes[t].nodes[n];
        break;
      }
    }
    if (src == NULL) continue;

    for (int l = 0; l < src->nlinks; l++) {
      struct ncclTopoLink* link = &src->links[l];
      if (link->type == LINK_LOC) continue;
      if (link->remNode == NULL) continue;
      if (link->bw <= 0) continue;
      int remKind = blinkVtxKindFromType(link->remNode->type);
      if (remKind < 0) continue;          // skip links to GIN/etc.
      int remIdx = blinkPhysFindVertex(pg, remKind, link->remNode->id);
      if (remIdx < 0) continue;           // remote not enumerated
      if (blinkPhysAddEdge(pg, v, remIdx, link->bw, link->type) < 0) {
        WARN("Blink: BLINK_MAX_PHYS_EDGES (%d) exceeded",
             BLINK_MAX_PHYS_EDGES);
        return ncclInternalError;
      }
    }
  }

  INFO(NCCL_GRAPH, "Blink phys graph: %d vertices (%d GPUs), %d edges",
       pg->nVertices, pg->nGpus, pg->nEdges);

  return ncclSuccess;
}

/*------------------------------------------------------------------------
 * Inter-host fabric (M4): add a synthetic FABRIC vertex and connect every
 * NET node to it (NET<->FABRIC bidirectional) at the NET node's line
 * rate. The user's model: NICs/NETs talk to each other through a single
 * black-box switch with per-NIC egress caps. The cap is automatically
 * enforced because each NET has only one outgoing edge to FABRIC, sized
 * to its line rate.
 *
 * This is a no-op when no NET nodes exist (single-host PCIe-only).
 *------------------------------------------------------------------------*/
static ncclResult_t blinkAddFabric(struct BlinkPhysGraph* pg) {
  // Sentinel topoId for the FABRIC vertex.
  static const int64_t FABRIC_TOPO_ID = -1;

  bool anyNet = false;
  for (int v = 0; v < pg->nVertices; v++) {
    if (pg->vertices[v].kind == BLINK_VTX_NET) { anyNet = true; break; }
  }
  if (!anyNet) return ncclSuccess;

  // If FABRIC already exists (e.g. a stale post-merge graph), no-op.
  for (int v = 0; v < pg->nVertices; v++) {
    if (pg->vertices[v].kind == BLINK_VTX_FABRIC) return ncclSuccess;
  }

  int fabricIdx = blinkPhysAddVertex(pg, BLINK_VTX_FABRIC, -1, FABRIC_TOPO_ID, -1);
  if (fabricIdx < 0) {
    WARN("Blink: BLINK_MAX_PHYS_VERTICES exceeded adding FABRIC");
    return ncclInternalError;
  }

  int added = 0;
  for (int v = 0; v < pg->nVertices; v++) {
    if (pg->vertices[v].kind != BLINK_VTX_NET) continue;
    float bw = pg->vertices[v].bw;
    if (bw <= 0) continue;
    if (blinkPhysAddEdge(pg, v, fabricIdx, bw, BLINK_LINK_FAB) < 0 ||
        blinkPhysAddEdge(pg, fabricIdx, v, bw, BLINK_LINK_FAB) < 0) {
      WARN("Blink: BLINK_MAX_PHYS_EDGES exceeded adding NET<->FABRIC");
      return ncclInternalError;
    }
    added++;
  }

  INFO(NCCL_GRAPH, "Blink: added FABRIC vertex with %d NET<->FABRIC pairs",
       added);
  return ncclSuccess;
}

/*------------------------------------------------------------------------
 * Cross-host topology aggregation (M4).
 *
 * Each rank serializes its local-host BlinkPhysGraph (intra-host only,
 * no FABRIC yet) and bootstrap-allgathers across all ranks. Then each
 * rank reconstructs a single global graph by stitching every distinct
 * host's vertices/edges together; FABRIC + NET<->FABRIC edges are added
 * once at the end. Computation is fully replicated -- MWU is
 * deterministic, so no broadcast is needed.
 *
 * Wire-format: a fixed-size BlinkPhysGraph blob per rank. Same-host
 * ranks emit identical blobs; on the receive side we keep one slot per
 * unique hostId.
 *------------------------------------------------------------------------*/
static ncclResult_t blinkExchangeTopology(struct ncclComm* comm,
                                          const struct BlinkPhysGraph* localPg,
                                          struct BlinkPhysGraph* globalOut) {
  if (comm == NULL) {
    memcpy(globalOut, localPg, sizeof(struct BlinkPhysGraph));
    return ncclSuccess;
  }
  int nRanks = comm->nRanks;
  int rank = comm->rank;
  if (nRanks <= 1) {
    memcpy(globalOut, localPg, sizeof(struct BlinkPhysGraph));
    return ncclSuccess;
  }

  size_t per = sizeof(struct BlinkPhysGraph);
  struct BlinkPhysGraph* slots = NULL;
  ncclResult_t ret = ncclSuccess;
  int seenHosts[BLINK_MAX_PHYS_VERTICES];
  int nSeenHosts = 0;
  int nGlobalGpus = 0;
  NCCLCHECKGOTO(ncclCalloc(&slots, nRanks), ret, cleanup);
  memcpy(&slots[rank], localPg, per);

  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, slots, (int)per), ret, cleanup);

  // Merge: walk each peer slot, dedup by hostId, append vertices and edges
  // into globalOut. Re-base edge endpoints into the global vertex index
  // space using a per-slot remap table.
  memset(globalOut, 0, sizeof(struct BlinkPhysGraph));

  for (int r = 0; r < nRanks; r++) {
    const struct BlinkPhysGraph* peer = &slots[r];
    if (peer->nVertices == 0) continue;

    // Skip if we've already absorbed a peer with this hostId (every rank
    // on the same host produces an identical blob).
    int peerHost = -1;
    for (int v = 0; v < peer->nVertices; v++) {
      if (peer->vertices[v].kind != BLINK_VTX_FABRIC) {
        peerHost = peer->vertices[v].hostId;
        break;
      }
    }
    if (peerHost < 0) continue;
    bool dup = false;
    for (int k = 0; k < nSeenHosts; k++) if (seenHosts[k] == peerHost) { dup = true; break; }
    if (dup) continue;
    seenHosts[nSeenHosts++] = peerHost;

    int remap[BLINK_MAX_PHYS_VERTICES];
    for (int v = 0; v < peer->nVertices; v++) remap[v] = -1;

    for (int v = 0; v < peer->nVertices; v++) {
      const struct BlinkPhysVertex* pv = &peer->vertices[v];
      if (pv->kind == BLINK_VTX_FABRIC) continue;  // FABRIC is added later
      int idx = blinkPhysAddVertex(globalOut, pv->kind, pv->hostId,
                                   pv->topoId, pv->gpuLocalIdx);
      if (idx < 0) {
        WARN("Blink: BLINK_MAX_PHYS_VERTICES exceeded during merge");
        ret = ncclInternalError; goto cleanup;
      }
      globalOut->vertices[idx].rank = pv->rank;
      globalOut->vertices[idx].bw = pv->bw;
      remap[v] = idx;
      if (pv->kind == BLINK_VTX_GPU) {
        if (nGlobalGpus >= BLINK_MAX_GPUS) {
          WARN("Blink: BLINK_MAX_GPUS=%d exceeded during cross-host merge",
               BLINK_MAX_GPUS);
          ret = ncclInternalError; goto cleanup;
        }
        globalOut->gpuVtxIdx[nGlobalGpus] = idx;
        globalOut->gpuRanks[nGlobalGpus] = pv->rank;
        // Reassign gpuLocalIdx so it indexes into the global gpu list.
        globalOut->vertices[idx].gpuLocalIdx = nGlobalGpus;
        nGlobalGpus++;
      }
    }

    for (int e = 0; e < peer->nEdges; e++) {
      const struct BlinkPhysEdge* pe = &peer->edges[e];
      int gs = (pe->src >= 0 && pe->src < peer->nVertices) ? remap[pe->src] : -1;
      int gd = (pe->dst >= 0 && pe->dst < peer->nVertices) ? remap[pe->dst] : -1;
      if (gs < 0 || gd < 0) continue;   // skipped FABRIC, etc.
      if (blinkPhysAddEdge(globalOut, gs, gd, pe->capacity, pe->kind) < 0) {
        WARN("Blink: BLINK_MAX_PHYS_EDGES exceeded during merge");
        ret = ncclInternalError; goto cleanup;
      }
    }
  }
  globalOut->nGpus = nGlobalGpus;

  INFO(NCCL_GRAPH,
       "Blink global merge: %d unique hosts, %d vertices, %d edges, %d GPUs",
       nSeenHosts, globalOut->nVertices, globalOut->nEdges, globalOut->nGpus);

cleanup:
  if (slots) free(slots);
  return ret;
}

/*------------------------------------------------------------------------
 * Bottleneck-path precomputation (M2).
 *
 * For every ordered (GPU_i, GPU_j) pair, find the path through the
 * physical topology that maximises the minimum edge capacity (max-min /
 * widest-path). Tie-break by hop count to prefer shorter paths.
 *
 * The result feeds two consumers:
 *   1. blinkBuildLogicalGraph -- creates a GPU-only `BlinkGraph` whose
 *      edges (i,j) carry capacity = bottleneck[i][j], for MWU/Edmonds.
 *   2. blinkRefinePhys        -- uses the path edge lists to decrement
 *      residuals on physical edges as trees are admitted, so shared
 *      switches/NICs are not over-packed.
 *
 * Complexity: O(nGpus * V^2) using the simple priority-free Dijkstra
 * variant. With V up to ~64 vertices and nGpus up to 16, this is well
 * under a millisecond per call.
 *------------------------------------------------------------------------*/

static ncclResult_t blinkComputeBottleneckPaths(const struct BlinkPhysGraph* pg,
                                                struct BlinkPathTable* pt) {
  memset(pt, 0, sizeof(struct BlinkPathTable));

  int V = pg->nVertices;
  int nGpus = pg->nGpus;

  // Per-source scratch
  float dist[BLINK_MAX_PHYS_VERTICES];
  int hops[BLINK_MAX_PHYS_VERTICES];
  int parentEdge[BLINK_MAX_PHYS_VERTICES];
  int finalized[BLINK_MAX_PHYS_VERTICES];

  for (int srcGpu = 0; srcGpu < nGpus; srcGpu++) {
    int srcVtx = pg->gpuVtxIdx[srcGpu];

    for (int v = 0; v < V; v++) {
      dist[v] = 0.0f;
      hops[v] = INT32_MAX;
      parentEdge[v] = -1;
      finalized[v] = 0;
    }
    dist[srcVtx] = FLT_MAX;
    hops[srcVtx] = 0;

    for (int iter = 0; iter < V; iter++) {
      // Pick unfinalized vertex with maximum dist (ties: minimum hops).
      int u = -1;
      float bestDist = -1.0f;
      int bestHops = INT32_MAX;
      for (int v = 0; v < V; v++) {
        if (finalized[v]) continue;
        if (dist[v] > bestDist ||
            (dist[v] == bestDist && hops[v] < bestHops)) {
          u = v;
          bestDist = dist[v];
          bestHops = hops[v];
        }
      }
      if (u < 0 || dist[u] <= 0.0f) break;
      finalized[u] = 1;

      // Relax outgoing edges from u.
      for (int e = 0; e < pg->nEdges; e++) {
        if (pg->edges[e].src != u) continue;
        int w = pg->edges[e].dst;
        if (finalized[w]) continue;
        float cap = pg->edges[e].capacity;
        float cand = (cap < dist[u]) ? cap : dist[u];
        int candHops = hops[u] + 1;
        if (cand > dist[w] ||
            (cand == dist[w] && candHops < hops[w])) {
          dist[w] = cand;
          hops[w] = candHops;
          parentEdge[w] = e;
        }
      }
    }

    // Reconstruct path to every other GPU.
    for (int dstGpu = 0; dstGpu < nGpus; dstGpu++) {
      if (dstGpu == srcGpu) continue;
      int dstVtx = pg->gpuVtxIdx[dstGpu];
      struct BlinkPath* path = &pt->paths[srcGpu][dstGpu];
      path->nHops = 0;
      path->bottleneck = 0.0f;
      if (dist[dstVtx] <= 0.0f) continue;   // unreachable

      // Walk back parentEdge chain; we get hops in reverse order.
      int rev[BLINK_MAX_PATH_HOPS];
      int n = 0;
      int cur = dstVtx;
      while (cur != srcVtx) {
        if (n >= BLINK_MAX_PATH_HOPS) {
          WARN("Blink: path src=%d dst=%d exceeds BLINK_MAX_PATH_HOPS=%d",
               srcGpu, dstGpu, BLINK_MAX_PATH_HOPS);
          n = 0;
          break;
        }
        int e = parentEdge[cur];
        if (e < 0) { n = 0; break; }
        rev[n++] = e;
        cur = pg->edges[e].src;
      }
      if (n == 0) continue;

      path->nHops = n;
      path->bottleneck = dist[dstVtx];
      for (int i = 0; i < n; i++) path->edges[i] = rev[n - 1 - i];
    }
  }

  return ncclSuccess;
}

// Convert the path table into a GPU-only BlinkGraph that the existing
// MWU/Edmonds code consumes. Logical edge (i,j) capacity = bottleneck[i][j].
// Pairs with no path are omitted.
static ncclResult_t blinkBuildLogicalGraph(const struct BlinkPhysGraph* pg,
                                           const struct BlinkPathTable* pt,
                                           struct BlinkGraph* bg) {
  memset(bg, 0, sizeof(struct BlinkGraph));
  memset(bg->edgeIndex, -1, sizeof(bg->edgeIndex));
  int nGpus = pg->nGpus;
  if (nGpus > BLINK_MAX_GPUS) {
    WARN("Blink: too many GPUs (%d > %d)", nGpus, BLINK_MAX_GPUS);
    return ncclInternalError;
  }
  bg->nVertices = nGpus;
  for (int i = 0; i < nGpus; i++) bg->gpuRanks[i] = pg->gpuRanks[i];

  for (int i = 0; i < nGpus; i++) {
    for (int j = 0; j < nGpus; j++) {
      if (i == j) continue;
      const struct BlinkPath* p = &pt->paths[i][j];
      if (p->nHops == 0 || p->bottleneck <= 0) continue;
      if (bg->nEdges >= BLINK_MAX_EDGES) {
        WARN("Blink: too many logical edges");
        return ncclInternalError;
      }
      bg->edgeIndex[i][j] = bg->nEdges;
      bg->edges[bg->nEdges].src = i;
      bg->edges[bg->nEdges].dst = j;
      bg->edges[bg->nEdges].capacity = p->bottleneck;
      bg->nEdges++;
    }
  }
  INFO(NCCL_GRAPH, "Blink logical graph: %d GPUs, %d edges (from physical paths)",
       bg->nVertices, bg->nEdges);
  return ncclSuccess;
}

// For a tree in logical-graph space, build the physical-edge usage
// histogram: usage[e] = number of logical tree edges whose path contains e.
// Returns -1 if any tree edge has no path (disconnected in physical graph).
static int blinkTreeBuildUsage(const struct BlinkGraph* bg,
                               const struct BlinkTree* tree,
                               const struct BlinkPathTable* pt,
                               int* usage,
                               int nPhysEdges) {
  memset(usage, 0, sizeof(int) * nPhysEdges);
  for (int v = 0; v < bg->nVertices; v++) {
    if (tree->parent[v] < 0) continue;
    int s = tree->parent[v], d = v;
    const struct BlinkPath* p = &pt->paths[s][d];
    if (p->nHops == 0) return -1;
    for (int h = 0; h < p->nHops; h++) {
      int e = p->edges[h];
      if (e < 0 || e >= nPhysEdges) return -1;
      usage[e]++;
    }
  }
  return 0;
}

// Effective per-tree bottleneck on physical residuals: min over used phys
// edges of (residual[e] / usage[e]). Captures the case where the same
// physical edge is shared by multiple logical edges of the same tree.
static float blinkTreePhysBottleneck(const int* usage,
                                     const float* residual,
                                     int nPhysEdges,
                                     bool* outFeasible) {
  float bottleneck = FLT_MAX;
  bool any = false;
  for (int e = 0; e < nPhysEdges; e++) {
    if (usage[e] == 0) continue;
    any = true;
    if (residual[e] <= 0) {
      *outFeasible = false;
      return 0.0f;
    }
    float capPerUse = residual[e] / (float)usage[e];
    if (capPerUse < bottleneck) bottleneck = capPerUse;
  }
  if (!any) {
    *outFeasible = false;
    return 0.0f;
  }
  *outFeasible = true;
  return bottleneck;
}

static void blinkTreeApplyToResidual(const int* usage, float* residual,
                                     int nPhysEdges, float w) {
  for (int e = 0; e < nPhysEdges; e++) {
    if (usage[e] == 0) continue;
    residual[e] -= w * (float)usage[e];
    if (residual[e] < 0) residual[e] = 0;
  }
}

/*========================================================================
 * Section 2: MWU (Multiplicative Weight Update) Tree Packing
 *========================================================================*/

// Compute total weight of a tree under given edge weights
static double blinkTreeWeight(const struct BlinkGraph* bg, const struct BlinkTree* tree,
                              const double* edgeWeights) {
  double total = 0.0;
  for (int v = 0; v < bg->nVertices; v++) {
    if (tree->parent[v] < 0) continue;
    int e = bg->edgeIndex[tree->parent[v]][v];
    if (e >= 0) total += edgeWeights[e];
  }
  return total;
}

// Compute bottleneck capacity of a tree (minimum edge capacity along tree edges)
static float blinkTreeBottleneck(const struct BlinkGraph* bg, const struct BlinkTree* tree) {
  float bottleneck = FLT_MAX;
  for (int v = 0; v < bg->nVertices; v++) {
    if (tree->parent[v] < 0) continue;
    int e = bg->edgeIndex[tree->parent[v]][v];
    if (e >= 0 && bg->edges[e].capacity < bottleneck) {
      bottleneck = bg->edges[e].capacity;
    }
  }
  return bottleneck;
}

/*------------------------------------------------------------------------
 * JSON dump (M3).
 *
 * Triggered when NCCL_BLINK_DUMP_JSON=<path> is set. Emits the physical
 * topology + spanning trees in a layout the viewer in
 * nccl-blink/contrib/blink-viewer/ can render. The file is rewritten on
 * every call (single-rank, single-shot dump). Format documented inline.
 *
 * Vertex IDs are stable strings of the form "<host>/<kind>/<localId>".
 * They are referenced verbatim by edges and channel tree_edges so the
 * viewer can use them as Cytoscape node IDs.
 *------------------------------------------------------------------------*/

static void blinkVtxJsonId(const struct BlinkPhysVertex* vtx, char* out, size_t outSz) {
  if (vtx->kind == BLINK_VTX_FABRIC) {
    snprintf(out, outSz, "fabric");
    return;
  }
  int64_t localId = NCCL_TOPO_ID_LOCAL_ID(vtx->topoId);
  snprintf(out, outSz, "%d/%s/%lx",
           vtx->hostId, blinkVtxKindStr(vtx->kind), (long)localId);
}

static const char* blinkLinkKindStr(int kind) {
  switch (kind) {
    case LINK_LOC: return "LOC";
    case LINK_NVL: return "NVL";
    case LINK_C2C: return "C2C";
    case LINK_PCI: return "PCI";
    case LINK_SYS: return "SYS";
    case LINK_NET: return "NET";
    case BLINK_LINK_FAB: return "FAB";
    default:       return "OTH";
  }
}

static ncclResult_t blinkDumpJson(const struct BlinkPhysGraph* pg,
                                  const struct BlinkPathTable* pt,
                                  const struct BlinkGraph* bg,
                                  const struct BlinkPacking* packing,
                                  const char* path) {
  FILE* f = fopen(path, "w");
  if (f == NULL) {
    WARN("Blink: NCCL_BLINK_DUMP_JSON: cannot open %s for writing", path);
    return ncclSystemError;
  }

  // hosts: discovered as the set of hostIds present on non-FABRIC vertices.
  int hostIds[BLINK_MAX_PHYS_VERTICES];
  int nHosts = 0;
  for (int v = 0; v < pg->nVertices; v++) {
    if (pg->vertices[v].kind == BLINK_VTX_FABRIC) continue;
    int h = pg->vertices[v].hostId;
    bool seen = false;
    for (int k = 0; k < nHosts; k++) if (hostIds[k] == h) { seen = true; break; }
    if (!seen) hostIds[nHosts++] = h;
  }

  fprintf(f, "{\n");
  fprintf(f, "  \"hosts\": [");
  for (int k = 0; k < nHosts; k++) {
    fprintf(f, "%s{\"systemId\": %d}", k == 0 ? "" : ", ", hostIds[k]);
  }
  fprintf(f, "],\n");

  // nodes
  fprintf(f, "  \"nodes\": [\n");
  for (int v = 0; v < pg->nVertices; v++) {
    char id[64];
    blinkVtxJsonId(&pg->vertices[v], id, sizeof(id));
    fprintf(f, "    {\"id\": \"%s\", \"type\": \"%s\", \"host\": %d",
            id, blinkVtxKindStr(pg->vertices[v].kind), pg->vertices[v].hostId);
    if (pg->vertices[v].kind == BLINK_VTX_GPU) {
      int g = pg->vertices[v].gpuLocalIdx;
      fprintf(f, ", \"rank\": %d, \"gpuLocal\": %d", pg->gpuRanks[g], g);
    }
    fprintf(f, "}%s\n", v + 1 < pg->nVertices ? "," : "");
  }
  fprintf(f, "  ],\n");

  // edges
  fprintf(f, "  \"edges\": [\n");
  for (int e = 0; e < pg->nEdges; e++) {
    char src[64], dst[64];
    blinkVtxJsonId(&pg->vertices[pg->edges[e].src], src, sizeof(src));
    blinkVtxJsonId(&pg->vertices[pg->edges[e].dst], dst, sizeof(dst));
    fprintf(f, "    {\"src\": \"%s\", \"dst\": \"%s\", \"type\": \"%s\", \"bw\": %.3f}%s\n",
            src, dst, blinkLinkKindStr(pg->edges[e].kind),
            pg->edges[e].capacity, e + 1 < pg->nEdges ? "," : "");
  }
  fprintf(f, "  ],\n");

  // channels: for each tree, dump the per-(parent,child) logical edge along
  // with its physical-path projection.
  fprintf(f, "  \"channels\": [\n");
  for (int c = 0; c < packing->nTrees; c++) {
    const struct BlinkTree* tree = &packing->trees[c];
    char rootId[64];
    int rootVtx = pg->gpuVtxIdx[tree->root];
    blinkVtxJsonId(&pg->vertices[rootVtx], rootId, sizeof(rootId));
    fprintf(f, "    {\"id\": %d, \"root\": \"%s\", \"weight\": %.3f, \"tree_edges\": [\n",
            c, rootId, tree->weight);
    bool first = true;
    for (int v = 0; v < bg->nVertices; v++) {
      if (tree->parent[v] < 0) continue;
      int s = tree->parent[v], d = v;
      char sId[64], dId[64];
      blinkVtxJsonId(&pg->vertices[pg->gpuVtxIdx[s]], sId, sizeof(sId));
      blinkVtxJsonId(&pg->vertices[pg->gpuVtxIdx[d]], dId, sizeof(dId));
      const struct BlinkPath* p = &pt->paths[s][d];
      fprintf(f, "      %s{\"src\": \"%s\", \"dst\": \"%s\", \"path\": [",
              first ? "" : ",", sId, dId);
      // The path begins at s (implicit) and ends at d (implicit). We emit
      // the full sequence of vertex IDs the path traverses, which the
      // viewer overlays as a polyline.
      char vId[64];
      blinkVtxJsonId(&pg->vertices[pg->gpuVtxIdx[s]], vId, sizeof(vId));
      fprintf(f, "\"%s\"", vId);
      for (int h = 0; h < p->nHops; h++) {
        int dstVtx = pg->edges[p->edges[h]].dst;
        blinkVtxJsonId(&pg->vertices[dstVtx], vId, sizeof(vId));
        fprintf(f, ", \"%s\"", vId);
      }
      fprintf(f, "]}\n");
      first = false;
    }
    fprintf(f, "    ]}%s\n", c + 1 < packing->nTrees ? "," : "");
  }
  fprintf(f, "  ]\n");
  fprintf(f, "}\n");
  fclose(f);

  INFO(NCCL_GRAPH, "Blink: dumped JSON topology+trees to %s", path);
  return ncclSuccess;
}

static ncclResult_t blinkMWU(const struct BlinkGraph* bg, double epsilon,
                             struct BlinkPacking* packing) {
  int V = bg->nVertices;
  int E = bg->nEdges;

  memset(packing, 0, sizeof(struct BlinkPacking));

  if (V <= 1 || E == 0) return ncclSuccess;

  // Initialize edge weights: w[e] = 1 / capacity[e]
  double w[BLINK_MAX_EDGES];
  for (int e = 0; e < E; e++) {
    w[e] = 1.0 / (double)bg->edges[e].capacity;
  }

  int maxIter = (int)ceil(log((double)E) / (epsilon * epsilon));
  if (maxIter > 1000) maxIter = 1000;
  if (maxIter < 10) maxIter = 10;

  INFO(NCCL_GRAPH, "Blink MWU: %d vertices, %d edges, epsilon=%.2f, maxIter=%d",
       V, E, epsilon, maxIter);

  for (int iter = 0; iter < maxIter; iter++) {
    // Try each vertex as root, find the best min-weight arborescence
    struct BlinkTree bestTree;
    double bestWeight = DBL_MAX;
    int bestRoot = -1;

    for (int r = 0; r < V; r++) {
      struct BlinkTree candidate;
      candidate.root = r;
      candidate.weight = 0;
      ncclResult_t ret = blinkEdmonds(bg, r, w, &candidate);
      if (ret != ncclSuccess) continue; // disconnected from this root

      double tw = blinkTreeWeight(bg, &candidate, w);
      if (tw < bestWeight) {
        bestWeight = tw;
        memcpy(&bestTree, &candidate, sizeof(struct BlinkTree));
        bestRoot = r;
      }
    }

    if (bestRoot < 0) break;

    // Record this tree. Weights are not meaningful here; blinkRefine
    // recomputes them via greedy bottleneck selection.
    if (packing->nTrees < BLINK_MAX_TREES) {
      memcpy(&packing->trees[packing->nTrees], &bestTree, sizeof(struct BlinkTree));
      packing->nTrees++;
    }

    // Update edge weights multiplicatively
    for (int v = 0; v < V; v++) {
      if (bestTree.parent[v] < 0) continue;
      int e = bg->edgeIndex[bestTree.parent[v]][v];
      if (e >= 0) {
        w[e] *= (1.0 + epsilon / (double)bg->edges[e].capacity);
      }
    }
  }

  INFO(NCCL_GRAPH, "Blink MWU: found %d candidate trees", packing->nTrees);
  return ncclSuccess;
}

/*========================================================================
 * Section 3: Greedy Refinement
 *========================================================================*/

static int blinkTreesEqual(const struct BlinkTree* a, const struct BlinkTree* b, int nVerts) {
  return a->root == b->root && memcmp(a->parent, b->parent, sizeof(int) * nVerts) == 0;
}

static ncclResult_t blinkRefine(const struct BlinkGraph* bg,
                                struct BlinkPacking* packing,
                                int targetCount) {
  int V = bg->nVertices;
  int E = bg->nEdges;

  if (packing->nTrees == 0 || targetCount <= 0) return ncclSuccess;

  // Deduplicate trees
  int uniqueCount = 0;
  struct BlinkTree unique[BLINK_MAX_TREES];
  int occurrence[BLINK_MAX_TREES];
  memset(occurrence, 0, sizeof(occurrence));

  for (int t = 0; t < packing->nTrees; t++) {
    int found = -1;
    for (int u = 0; u < uniqueCount; u++) {
      if (blinkTreesEqual(&packing->trees[t], &unique[u], V)) {
        found = u;
        break;
      }
    }
    if (found >= 0) {
      occurrence[found]++;
    } else if (uniqueCount < BLINK_MAX_TREES) {
      memcpy(&unique[uniqueCount], &packing->trees[t], sizeof(struct BlinkTree));
      occurrence[uniqueCount] = 1;
      uniqueCount++;
    }
  }

  INFO(NCCL_GRAPH, "Blink refine: %d unique trees from %d candidates, target=%d channels",
       uniqueCount, packing->nTrees, targetCount);

  // Greedy selection on residual capacities
  float residual[BLINK_MAX_EDGES];
  for (int e = 0; e < E; e++) {
    residual[e] = bg->edges[e].capacity;
  }

  struct BlinkTree selected[BLINK_MAX_TREES];
  int nSelected = 0;

  for (int round = 0; round < targetCount && round < BLINK_MAX_TREES; round++) {
    int bestIdx = -1;
    float bestBw = 0;

    for (int u = 0; u < uniqueCount; u++) {
      float bw = FLT_MAX;
      bool feasible = true;
      for (int v = 0; v < V; v++) {
        if (unique[u].parent[v] < 0) continue;
        int e = bg->edgeIndex[unique[u].parent[v]][v];
        if (e < 0) { feasible = false; break; }
        if (residual[e] < bg->edges[e].capacity * 1e-4f) {
          feasible = false; break;
        } else if (residual[e] < bw) {
          bw = residual[e];
        }
      }
      if (feasible && bw > bestBw) {
        bestBw = bw;
        bestIdx = u;
      }
    }

    if (bestIdx < 0) break;  // no more feasible trees

    memcpy(&selected[nSelected], &unique[bestIdx], sizeof(struct BlinkTree));
    selected[nSelected].weight = bestBw;
    nSelected++;

    // Subtract usage from residual
    for (int v = 0; v < V; v++) {
      if (unique[bestIdx].parent[v] < 0) continue;
      int e = bg->edgeIndex[unique[bestIdx].parent[v]][v];
      if (e >= 0) {
        residual[e] -= bestBw;
        if (residual[e] < 0) residual[e] = 0;
      }
    }
  }

  if (nSelected == 0) {
    WARN("Blink: refine found no feasible trees");
    return ncclInternalError;
  }

  // Fill remaining channels by duplicating with zero weight
  if (nSelected < targetCount) {
    INFO(NCCL_GRAPH, "Blink refine: padding %d zero-weight channels (had %d, need %d)",
         targetCount - nSelected, nSelected, targetCount);
    int base = nSelected;
    while (nSelected < targetCount && nSelected < BLINK_MAX_TREES) {
      memcpy(&selected[nSelected], &selected[nSelected % base], sizeof(struct BlinkTree));
      selected[nSelected].weight = 0;
      nSelected++;
    }
  }

  packing->nTrees = nSelected;
  packing->totalRate = 0;
  for (int t = 0; t < nSelected; t++) {
    memcpy(&packing->trees[t], &selected[t], sizeof(struct BlinkTree));
    packing->totalRate += selected[t].weight;
  }

  INFO(NCCL_GRAPH, "Blink refine: selected %d trees, total rate=%.2f GB/s",
       packing->nTrees, packing->totalRate);
  return ncclSuccess;
}

/*------------------------------------------------------------------------
 * Physical-residual refinement (M2).
 *
 * Same shape as `blinkRefine` -- dedup, then greedy bottleneck pick --
 * but residuals live on physical edges, and "tree uses logical edge
 * (i,j)" means "decrement every physical edge in pt->paths[i][j]" by
 * the tree's admitted weight.
 *------------------------------------------------------------------------*/
static ncclResult_t blinkRefinePhys(const struct BlinkGraph* bg,
                                    const struct BlinkPhysGraph* pg,
                                    const struct BlinkPathTable* pt,
                                    struct BlinkPacking* packing,
                                    int targetCount) {
  int V = bg->nVertices;
  int E = pg->nEdges;
  if (packing->nTrees == 0 || targetCount <= 0) return ncclSuccess;

  int uniqueCount = 0;
  struct BlinkTree unique[BLINK_MAX_TREES];
  int occurrence[BLINK_MAX_TREES];
  memset(occurrence, 0, sizeof(occurrence));

  for (int t = 0; t < packing->nTrees; t++) {
    int found = -1;
    for (int u = 0; u < uniqueCount; u++) {
      if (blinkTreesEqual(&packing->trees[t], &unique[u], V)) {
        found = u;
        break;
      }
    }
    if (found >= 0) {
      occurrence[found]++;
    } else if (uniqueCount < BLINK_MAX_TREES) {
      memcpy(&unique[uniqueCount], &packing->trees[t], sizeof(struct BlinkTree));
      occurrence[uniqueCount] = 1;
      uniqueCount++;
    }
  }

  INFO(NCCL_GRAPH, "Blink phys-refine: %d unique trees from %d candidates, target=%d",
       uniqueCount, packing->nTrees, targetCount);

  float residual[BLINK_MAX_PHYS_EDGES];
  for (int e = 0; e < E; e++) residual[e] = pg->edges[e].capacity;

  struct BlinkTree selected[BLINK_MAX_TREES];
  int nSelected = 0;

  int curUsage[BLINK_MAX_PHYS_EDGES];
  int bestUsage[BLINK_MAX_PHYS_EDGES];

  for (int round = 0; round < targetCount && round < BLINK_MAX_TREES; round++) {
    int bestIdx = -1;
    float bestBw = 0;
    memset(bestUsage, 0, sizeof(int) * E);

    for (int u = 0; u < uniqueCount; u++) {
      if (blinkTreeBuildUsage(bg, &unique[u], pt, curUsage, E) < 0) continue;
      bool feasible;
      float bw = blinkTreePhysBottleneck(curUsage, residual, E, &feasible);
      if (feasible && bw > bestBw) {
        bestBw = bw;
        bestIdx = u;
        memcpy(bestUsage, curUsage, sizeof(int) * E);
      }
    }

    if (bestIdx < 0) break;

    memcpy(&selected[nSelected], &unique[bestIdx], sizeof(struct BlinkTree));
    selected[nSelected].weight = bestBw;
    nSelected++;
    blinkTreeApplyToResidual(bestUsage, residual, E, bestBw);
  }

  if (nSelected == 0) {
    WARN("Blink phys-refine: found no feasible trees");
    return ncclInternalError;
  }

  if (nSelected < targetCount) {
    INFO(NCCL_GRAPH, "Blink phys-refine: padding %d zero-weight channels (had %d, need %d)",
         targetCount - nSelected, nSelected, targetCount);
    int base = nSelected;
    while (nSelected < targetCount && nSelected < BLINK_MAX_TREES) {
      memcpy(&selected[nSelected], &selected[nSelected % base], sizeof(struct BlinkTree));
      selected[nSelected].weight = 0;
      nSelected++;
    }
  }

  packing->nTrees = nSelected;
  packing->totalRate = 0;
  for (int t = 0; t < nSelected; t++) {
    memcpy(&packing->trees[t], &selected[t], sizeof(struct BlinkTree));
    packing->totalRate += selected[t].weight;
  }

  INFO(NCCL_GRAPH, "Blink phys-refine: selected %d trees, total rate=%.2f GB/s",
       packing->nTrees, packing->totalRate);
  return ncclSuccess;
}

/*========================================================================
 * Section 4: Tree-to-Chain Conversion
 *========================================================================*/

static ncclResult_t blinkTreeToChain(const struct BlinkTree* tree, int nVerts,
                                     const int* gpuRanks, int* chain) {
  int children[BLINK_MAX_GPUS][BLINK_MAX_GPUS];
  int nChildren[BLINK_MAX_GPUS];
  memset(nChildren, 0, sizeof(int) * nVerts);

  for (int v = 0; v < nVerts; v++) {
    if (tree->parent[v] >= 0 && tree->parent[v] != v) {
      int p = tree->parent[v];
      children[p][nChildren[p]++] = v;
    }
  }

  // DFS from root
  int stack[BLINK_MAX_GPUS];
  int stackTop = 0;
  stack[stackTop++] = tree->root;
  int pos = 0;

  while (stackTop > 0 && pos < nVerts) {
    int v = stack[--stackTop];
    chain[pos++] = gpuRanks[v];
    for (int c = nChildren[v] - 1; c >= 0; c--) {
      stack[stackTop++] = children[v][c];
    }
  }

  if (pos != nVerts) {
    WARN("Blink: DFS produced %d vertices, expected %d", pos, nVerts);
    return ncclInternalError;
  }

  return ncclSuccess;
}

/*========================================================================
 * Section 5: Main Entry Point
 *========================================================================*/

ncclResult_t ncclBlinkCompute(struct ncclComm* comm,
                              struct ncclTopoSystem* system,
                              struct ncclTopoGraph* graph,
                              int nChannelsNeeded) {
  int nGpus = system->nodes[GPU].count;
  (void)comm;  // referenced by blinkExchangeTopology when NCCL_BLINK_GLOBAL=1

  if (nGpus <= 1 || nChannelsNeeded <= 0) {
    INFO(NCCL_GRAPH, "Blink: nGpus=%d nChannels=%d, falling back to default",
         nGpus, nChannelsNeeded);
    graph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
    graph->minChannels = nChannelsNeeded;
    graph->maxChannels = nChannelsNeeded;
    return ncclTopoCompute(system, graph);
  }

  // When intra-host has < 4 GPUs and we are in a multinode (NIC-attached)
  // job, Blink's intra-host packing produces degenerate 1- or 2-edge trees
  // that NCCL stitches sub-optimally across hosts (empirically a 0.5x
  // multinode regression at 2 GPUs/host). Fall back to default in that
  // case until the multi-host MWU lands.
  if (nGpus < 4 && system->nodes[NET].count > 0) {
    INFO(NCCL_GRAPH, "Blink: nGpus=%d with nNets=%d, multinode degenerate; "
         "falling back to default", nGpus, system->nodes[NET].count);
    graph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
    graph->minChannels = nChannelsNeeded;
    graph->maxChannels = nChannelsNeeded;
    return ncclTopoCompute(system, graph);
  }

  struct timespec t0, t1, t2, t3, t4;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  // NCCL_BLINK_FULL_GRAPH=1 selects the M1+M2 pipeline: full physical
  // topology -> bottleneck-path precompute -> logical GPU graph fed into
  // existing MWU/Edmonds -> refinement that tracks residuals on physical
  // edges. Default off preserves the historical NVL-only behaviour.
  const char* fullGraphEnv = getenv("NCCL_BLINK_FULL_GRAPH");
  bool useFullGraph = (fullGraphEnv != NULL && fullGraphEnv[0] == '1');

  struct BlinkGraph bg;
  struct BlinkPhysGraph pg;
  struct BlinkPathTable pt;
  ncclResult_t ret;

  if (useFullGraph) {
    const char* globalEnv = getenv("NCCL_BLINK_GLOBAL");
    bool useGlobal = (comm != NULL && globalEnv != NULL && globalEnv[0] == '1');

    struct BlinkPhysGraph localPg;
    ret = blinkExtractFullGraph(system, &localPg);

    if (ret == ncclSuccess && useGlobal) {
      ret = blinkExchangeTopology(comm, &localPg, &pg);
    } else if (ret == ncclSuccess) {
      memcpy(&pg, &localPg, sizeof(struct BlinkPhysGraph));
    }
    if (ret == ncclSuccess) ret = blinkAddFabric(&pg);

    if (ret == ncclSuccess) {
      int kindCounts[7] = {0};
      for (int v = 0; v < pg.nVertices; v++) {
        if (pg.vertices[v].kind >= 0 && pg.vertices[v].kind < 7) {
          kindCounts[pg.vertices[v].kind]++;
        }
      }
      INFO(NCCL_GRAPH,
           "Blink phys%s: GPU=%d NVS=%d PCI=%d CPU=%d NIC=%d NET=%d FAB=%d edges=%d",
           useGlobal ? "(global)" : "",
           kindCounts[BLINK_VTX_GPU], kindCounts[BLINK_VTX_NVS],
           kindCounts[BLINK_VTX_PCI], kindCounts[BLINK_VTX_CPU],
           kindCounts[BLINK_VTX_NIC], kindCounts[BLINK_VTX_NET],
           kindCounts[BLINK_VTX_FABRIC], pg.nEdges);
    }
    if (ret == ncclSuccess) ret = blinkComputeBottleneckPaths(&pg, &pt);
    if (ret == ncclSuccess) ret = blinkBuildLogicalGraph(&pg, &pt, &bg);
    if (ret != ncclSuccess || bg.nEdges == 0) {
      INFO(NCCL_GRAPH, "Blink: full-graph build empty/failed, falling back to default");
      graph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
      graph->minChannels = nChannelsNeeded;
      graph->maxChannels = nChannelsNeeded;
      return ncclTopoCompute(system, graph);
    }
  } else {
    ret = blinkExtractGraph(system, &bg);
    if (ret != ncclSuccess || bg.nEdges == 0) {
      INFO(NCCL_GRAPH, "Blink: no NVLink edges, falling back to default");
      graph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
      graph->minChannels = nChannelsNeeded;
      graph->maxChannels = nChannelsNeeded;
      return ncclTopoCompute(system, graph);
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);

  // Step 2: Run MWU to find candidate trees on the logical graph.
  struct BlinkPacking packing;
  double epsilon = (bg.nVertices > 8) ? 0.05 : 0.1;
  // BLINK_EPSILON env override — used by ablations.
  const char* eps_env = getenv("BLINK_EPSILON");
  if (eps_env) {
    double v = atof(eps_env);
    if (v > 0.0 && v < 1.0) {
      epsilon = v;
      INFO(NCCL_GRAPH, "Blink: BLINK_EPSILON override = %.3f", epsilon);
    }
  }

  ret = blinkMWU(&bg, epsilon, &packing);
  clock_gettime(CLOCK_MONOTONIC, &t2);
  if (ret != ncclSuccess || packing.nTrees == 0) {
    INFO(NCCL_GRAPH, "Blink: MWU failed or found no trees, falling back");
    graph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
    graph->minChannels = nChannelsNeeded;
    graph->maxChannels = nChannelsNeeded;
    return ncclTopoCompute(system, graph);
  }

  // Step 3: Refine — select targetCount trees. With useFullGraph, residuals
  // are tracked on physical edges so shared switches/NICs are not
  // over-packed.
  if (useFullGraph) {
    ret = blinkRefinePhys(&bg, &pg, &pt, &packing, nChannelsNeeded);
  } else {
    ret = blinkRefine(&bg, &packing, nChannelsNeeded);
  }
  clock_gettime(CLOCK_MONOTONIC, &t3);
  if (ret != ncclSuccess || packing.nTrees == 0) {
    INFO(NCCL_GRAPH, "Blink: refinement failed, falling back");
    graph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
    graph->minChannels = nChannelsNeeded;
    graph->maxChannels = nChannelsNeeded;
    return ncclTopoCompute(system, graph);
  }

  // M3: optional JSON dump of the full physical topology + spanning
  // trees, consumed by nccl-blink/contrib/blink-viewer/. Only available on
  // the full-graph pipeline because it requires the physical graph and
  // path table.
  if (useFullGraph) {
    const char* dumpPath = getenv("NCCL_BLINK_DUMP_JSON");
    if (dumpPath != NULL && dumpPath[0] != '\0') {
      blinkDumpJson(&pg, &pt, &bg, &packing, dumpPath);
    }
  }

  // Step 4: Convert trees to ncclTopoGraph format. We keep TREE pattern
  // (rather than BALANCED_TREE) so NCCL's tuning code applies its 0.85x bw
  // multiplier to the projected Tree bandwidth, which is what nudges the
  // selector toward NVLS at large messages on asymmetric topologies. See
  // src/graph/tuning.cc:310.
  graph->id = 1;
  graph->pattern = NCCL_TOPO_PATTERN_TREE;
  graph->crossNic = 0;
  graph->collNet = 0;
  graph->nChannels = packing.nTrees;
  graph->sameChannels = 0;
  graph->nHops = nGpus - 1;
  graph->typeIntra = PATH_NVL;
  graph->typeInter = PATH_SYS;
  graph->latencyInter = 0;

  // Bandwidth: minimum bottleneck across trees with non-zero weight
  float minBw = FLT_MAX;
  int nonZeroTrees = 0;
  for (int t = 0; t < packing.nTrees; t++) {
    if (packing.trees[t].weight > 0) {
      float bw = blinkTreeBottleneck(&bg, &packing.trees[t]);
      if (bw < minBw) minBw = bw;
      nonZeroTrees++;
    }
  }
  if (nonZeroTrees == 0) minBw = 0;
  graph->bwIntra = minBw;
  graph->bwInter = 0;

  // For comms with NIC topology (single- or multi-host), select per-channel
  // NET IDs to populate graph->inter[]. For pure single-host with no NICs we
  // leave inter[] as -1 (sentinel meaning "no network hop"). This mirrors the
  // fallback path in search.cc:1218 which uses NET node count rather than
  // nHosts because each rank's local system view often reports nHosts=1 even
  // for multi-host jobs.
  bool hasNets = system->nodes[NET].count > 0;
  int64_t entryNetId = -1, exitNetId = -1;
  if (hasNets) {
    int nets[NCCL_TOPO_MAX_NODES];
    int netCount = 0;
    if (ncclTopoSelectNets(system, /*typeInter=*/-1, /*gpu=*/0,
                           nets, &netCount) == ncclSuccess && netCount > 0) {
      entryNetId = system->nodes[NET].nodes[nets[0]].id;
    }
    netCount = 0;
    if (ncclTopoSelectNets(system, /*typeInter=*/-1, /*gpu=*/nGpus-1,
                           nets, &netCount) == ncclSuccess && netCount > 0) {
      exitNetId = system->nodes[NET].nodes[nets[0]].id;
    }
    INFO(NCCL_GRAPH, "Blink: multi-host inter[] (nHosts=%d, nNets=%d) entry=%lx exit=%lx",
         system->nHosts, system->nodes[NET].count, (long)entryNetId, (long)exitNetId);
    if (entryNetId != -1 && exitNetId != -1) {
      graph->bwInter = 0.1f;  // placeholder — NCCL recomputes per-channel bw
      graph->typeInter = PATH_SYS;
    }
  }

  // Fill intra[] with DFS chain orderings; populate inter[] for multi-host.
  for (int c = 0; c < packing.nTrees; c++) {
    int chain[BLINK_MAX_GPUS];
    ret = blinkTreeToChain(&packing.trees[c], bg.nVertices, bg.gpuRanks, chain);
    if (ret != ncclSuccess) {
      INFO(NCCL_GRAPH, "Blink: chain conversion failed for tree %d, falling back", c);
      graph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
      graph->minChannels = nChannelsNeeded;
      graph->maxChannels = nChannelsNeeded;
      return ncclTopoCompute(system, graph);
    }
    for (int i = 0; i < bg.nVertices; i++) {
      graph->intra[c * nGpus + i] = chain[i];
    }
    graph->inter[c * 2 + 0] = entryNetId;
    graph->inter[c * 2 + 1] = exitNetId;
  }

  clock_gettime(CLOCK_MONOTONIC, &t4);

  INFO(NCCL_GRAPH, "Blink: computed %d tree channels, bwIntra=%.1f GB/s, totalRate=%.1f GB/s",
       graph->nChannels, graph->bwIntra, packing.totalRate);
  INFO(NCCL_GRAPH, "Blink timing: extract=%.1fus MWU=%.1fus refine=%.1fus chain=%.1fus total=%.1fus",
       blinkElapsedUs(&t0, &t1), blinkElapsedUs(&t1, &t2),
       blinkElapsedUs(&t2, &t3), blinkElapsedUs(&t3, &t4),
       blinkElapsedUs(&t0, &t4));

  for (int c = 0; c < graph->nChannels; c++) {
    char buf[256];
    int pos = 0;
    pos += snprintf(buf + pos, sizeof(buf) - pos, "Blink tree %d (root=%d, w=%.1f): ",
                    c, packing.trees[c].root, packing.trees[c].weight);
    for (int i = 0; i < nGpus && pos < (int)sizeof(buf) - 8; i++) {
      pos += snprintf(buf + pos, sizeof(buf) - pos, "%d ", graph->intra[c * nGpus + i]);
    }
    INFO(NCCL_GRAPH, "%s", buf);
  }

  return ncclSuccess;
}
