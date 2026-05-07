export type NodeType = 'GPU' | 'NVS' | 'PCI' | 'CPU' | 'NIC' | 'NET' | 'FAB';

export interface BlinkHost {
  systemId: number;
}

export interface BlinkNode {
  id: string;
  type: NodeType;
  host: number;
  rank?: number;
  gpuLocal?: number;
}

export interface BlinkEdge {
  src: string;
  dst: string;
  type: string;
  bw: number;
}

export interface BlinkTreeEdge {
  src: string;
  dst: string;
  path: string[];
}

export interface BlinkChannel {
  id: number;
  root: string;
  weight: number;
  tree_edges: BlinkTreeEdge[];
}

export interface BlinkDump {
  hosts: BlinkHost[];
  nodes: BlinkNode[];
  edges: BlinkEdge[];
  channels: BlinkChannel[];
}

export const NODE_COLORS: Record<NodeType, string> = {
  GPU: '#4caf50',
  NVS: '#9c27b0',
  PCI: '#2196f3',
  CPU: '#757575',
  NIC: '#ff9800',
  NET: '#f44336',
  FAB: '#ffc107',
};
