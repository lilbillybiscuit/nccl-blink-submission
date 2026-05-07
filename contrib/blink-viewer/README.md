# Blink Viewer

A small Vite + React + TypeScript SPA for visualizing the JSON dump produced
by the NCCL Blink topology engine. Renders the physical topology as a
Cytoscape graph and overlays the spanning tree for any selected channel.

## Run

```sh
cd contrib/blink-viewer
npm install
npm run dev
```

Open the printed URL (default `http://localhost:5173`), then drop a Blink
JSON dump onto the page (or use "Open JSON..." in the sidebar).

## Generate a JSON dump

Run any NCCL workload with the Blink debug-dump environment variables:

```sh
NCCL_BLINK=1 NCCL_BLINK_FULL_GRAPH=1 NCCL_BLINK_DUMP_JSON=/tmp/blink.json ./run
```

The producer is `blinkDumpJson` in `src/graph/blink.cc`.

## UI

- Sidebar lists every channel with its id, root, weight, and edge counts.
- Click a channel (or use arrow keys) to highlight its spanning tree.
- Number keys `0`-`9` jump to that channel id. `Esc` clears the selection.

Node colors:

| Type | Color   |
|------|---------|
| GPU  | green   |
| NVS  | purple  |
| PCI  | blue    |
| CPU  | gray    |
| NIC  | orange  |
| NET  | red     |
| FAB  | yellow  |

When a channel is selected, all physical edges that lie on any of its
`tree_edges[].path` sequences are drawn in red; everything else is dimmed.
The root GPU is outlined in gold and prefixed with stars.
