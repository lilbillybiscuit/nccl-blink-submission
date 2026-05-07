import { useEffect, useRef } from 'react';
import cytoscape, { Core, ElementDefinition } from 'cytoscape';
// @ts-expect-error - no type declaration shipped
import coseBilkent from 'cytoscape-cose-bilkent';
import { BlinkChannel, BlinkDump, NODE_COLORS } from './types';

let registered = false;
function ensureExtension() {
  if (!registered) {
    cytoscape.use(coseBilkent);
    registered = true;
  }
}

interface Props {
  dump: BlinkDump;
  selectedChannel: BlinkChannel | null;
}

// Build the set of physical edges traversed by a channel by walking each
// tree_edge.path and pairing consecutive vertices. Edge ID matches what we
// emit when constructing the Cytoscape graph below.
function physicalEdgesForChannel(channel: BlinkChannel): Set<string> {
  const set = new Set<string>();
  for (const te of channel.tree_edges) {
    for (let i = 0; i + 1 < te.path.length; i++) {
      set.add(edgeKey(te.path[i], te.path[i + 1]));
      // Edges in the dump are directed but a tree may traverse them either way;
      // mark both orientations so we light up the underlying physical edge.
      set.add(edgeKey(te.path[i + 1], te.path[i]));
    }
  }
  return set;
}

function physicalNodesForChannel(channel: BlinkChannel): Set<string> {
  const set = new Set<string>();
  for (const te of channel.tree_edges) {
    for (const v of te.path) set.add(v);
  }
  return set;
}

function edgeKey(src: string, dst: string): string {
  return `${src}__${dst}`;
}

export default function Topology({ dump, selectedChannel }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);

  // Build / rebuild the graph whenever the dump changes.
  useEffect(() => {
    ensureExtension();
    if (!containerRef.current) return;

    const elements: ElementDefinition[] = [];
    for (const n of dump.nodes) {
      let label = n.id;
      if (n.type === 'GPU' && n.rank !== undefined) {
        label = `GPU r${n.rank}`;
      } else if (n.type === 'FAB') {
        label = 'FABRIC';
      } else {
        // Drop the host prefix to keep labels short; keep the tail.
        const parts = n.id.split('/');
        label = parts.length >= 2 ? `${n.type} ${parts[parts.length - 1]}` : n.id;
      }
      elements.push({
        data: {
          id: n.id,
          label,
          ntype: n.type,
          host: n.host,
          color: NODE_COLORS[n.type],
        },
      });
    }
    for (const e of dump.edges) {
      elements.push({
        data: {
          id: edgeKey(e.src, e.dst),
          source: e.src,
          target: e.dst,
          label: `${e.type} ${e.bw.toFixed(1)}GB/s`,
          etype: e.type,
        },
      });
    }

    if (cyRef.current) {
      cyRef.current.destroy();
    }

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      wheelSensitivity: 0.2,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            label: 'data(label)',
            color: '#ffffff',
            'text-outline-color': '#000000',
            'text-outline-width': 2,
            'font-size': 10,
            'text-valign': 'center',
            'text-halign': 'center',
            width: 40,
            height: 40,
          },
        },
        {
          selector: 'node[ntype = "FAB"]',
          style: {
            shape: 'diamond',
            width: 60,
            height: 60,
          },
        },
        {
          selector: 'node[ntype = "GPU"]',
          style: {
            shape: 'round-rectangle',
            width: 50,
            height: 50,
          },
        },
        {
          selector: 'edge',
          style: {
            width: 2,
            'line-color': '#888',
            'target-arrow-color': '#888',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            label: 'data(label)',
            color: '#cccccc',
            'font-size': 8,
            'text-background-color': '#1e1e1e',
            'text-background-opacity': 0.7,
            'text-background-padding': '2px',
          },
        },
        {
          selector: '.dim',
          style: { opacity: 0.15 },
        },
        {
          selector: 'edge.tree',
          style: {
            'line-color': '#ff5252',
            'target-arrow-color': '#ff5252',
            width: 4,
            opacity: 1,
            'z-index': 999,
          },
        },
        {
          selector: 'node.tree',
          style: { 'border-width': 3, 'border-color': '#ff5252', opacity: 1 },
        },
        {
          selector: 'node.root',
          style: {
            'border-width': 4,
            'border-color': '#ffd54f',
            label: (ele: cytoscape.NodeSingular) => `* ${ele.data('label')} *`,
          },
        },
      ],
      layout: {
        name: 'cose-bilkent',
        // @ts-expect-error - extension-specific layout options
        nodeRepulsion: 8000,
        idealEdgeLength: 80,
        edgeElasticity: 0.45,
        gravity: 0.25,
        gravityRangeCompound: 1.5,
        numIter: 2500,
        tile: true,
        animate: false,
        randomize: true,
      },
    });

    cyRef.current = cy;
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [dump]);

  // Apply / clear the spanning-tree overlay whenever the selection changes.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements().removeClass('dim tree root');

    if (!selectedChannel) return;

    const treeEdges = physicalEdgesForChannel(selectedChannel);
    const treeNodes = physicalNodesForChannel(selectedChannel);

    cy.edges().forEach((e) => {
      const src = e.data('source');
      const dst = e.data('target');
      if (treeEdges.has(edgeKey(src, dst)) || treeEdges.has(edgeKey(dst, src))) {
        e.addClass('tree');
      } else {
        e.addClass('dim');
      }
    });

    cy.nodes().forEach((n) => {
      if (treeNodes.has(n.id())) {
        n.addClass('tree');
      } else {
        n.addClass('dim');
      }
    });

    const rootNode = cy.getElementById(selectedChannel.root);
    if (rootNode && rootNode.length) {
      rootNode.addClass('root');
    }
  }, [selectedChannel]);

  return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />;
}
