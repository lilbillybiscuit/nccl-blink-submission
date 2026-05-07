import { useCallback, useEffect, useMemo, useState } from 'react';
import Topology from './Topology';
import { BlinkChannel, BlinkDump, NODE_COLORS, NodeType } from './types';

function isBlinkDump(x: unknown): x is BlinkDump {
  if (!x || typeof x !== 'object') return false;
  const o = x as Record<string, unknown>;
  return Array.isArray(o.nodes) && Array.isArray(o.edges) && Array.isArray(o.channels);
}

export default function App() {
  const [dump, setDump] = useState<BlinkDump | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragHover, setDragHover] = useState(false);

  const loadFile = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onerror = () => setError('Failed to read file');
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result));
        if (!isBlinkDump(parsed)) {
          setError('JSON does not look like a Blink dump (missing nodes/edges/channels)');
          return;
        }
        setError(null);
        setDump(parsed);
        setSelectedId(parsed.channels.length ? parsed.channels[0].id : null);
      } catch (e) {
        setError(`Parse error: ${(e as Error).message}`);
      }
    };
    reader.readAsText(file);
  }, []);

  // Drag-and-drop on the document. Don't navigate when files are dropped
  // outside the explicit dropzone.
  useEffect(() => {
    const onDragOver = (ev: DragEvent) => {
      ev.preventDefault();
      setDragHover(true);
    };
    const onDragLeave = (ev: DragEvent) => {
      if (ev.target === document.documentElement) setDragHover(false);
    };
    const onDrop = (ev: DragEvent) => {
      ev.preventDefault();
      setDragHover(false);
      const f = ev.dataTransfer?.files?.[0];
      if (f) loadFile(f);
    };
    document.addEventListener('dragover', onDragOver);
    document.addEventListener('dragleave', onDragLeave);
    document.addEventListener('drop', onDrop);
    return () => {
      document.removeEventListener('dragover', onDragOver);
      document.removeEventListener('dragleave', onDragLeave);
      document.removeEventListener('drop', onDrop);
    };
  }, [loadFile]);

  const channels = dump?.channels ?? [];
  const selectedChannel: BlinkChannel | null = useMemo(
    () => channels.find((c) => c.id === selectedId) ?? null,
    [channels, selectedId],
  );

  // Keyboard navigation across channels.
  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if (!channels.length) return;
      if (ev.key === 'Escape') {
        setSelectedId(null);
        return;
      }
      if (ev.key === 'ArrowRight' || ev.key === 'ArrowLeft') {
        ev.preventDefault();
        const idx = channels.findIndex((c) => c.id === selectedId);
        const delta = ev.key === 'ArrowRight' ? 1 : -1;
        const next = idx === -1 ? 0 : (idx + delta + channels.length) % channels.length;
        setSelectedId(channels[next].id);
        return;
      }
      if (/^[0-9]$/.test(ev.key)) {
        const n = parseInt(ev.key, 10);
        const match = channels.find((c) => c.id === n);
        if (match) setSelectedId(match.id);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [channels, selectedId]);

  const onPickFile = (ev: React.ChangeEvent<HTMLInputElement>) => {
    const f = ev.target.files?.[0];
    if (f) loadFile(f);
  };

  const totalPathEdges = selectedChannel
    ? selectedChannel.tree_edges.reduce((acc, te) => acc + Math.max(0, te.path.length - 1), 0)
    : 0;

  return (
    <div style={{ display: 'flex', height: '100%' }}>
      <aside
        style={{
          width: 320,
          background: '#252525',
          borderRight: '1px solid #333',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <div style={{ padding: 12, borderBottom: '1px solid #333' }}>
          <h2 style={{ margin: 0, fontSize: 16 }}>NCCL Blink Viewer</h2>
          <div style={{ marginTop: 8, fontSize: 12, color: '#aaa' }}>
            {dump
              ? `${dump.nodes.length} nodes / ${dump.edges.length} edges / ${dump.channels.length} channels`
              : 'Drop a JSON dump or click below.'}
          </div>
          <label
            style={{
              display: 'inline-block',
              marginTop: 8,
              padding: '4px 8px',
              background: '#3a3a3a',
              border: '1px solid #555',
              borderRadius: 3,
              cursor: 'pointer',
              fontSize: 12,
            }}
          >
            Open JSON...
            <input
              type="file"
              accept="application/json,.json"
              onChange={onPickFile}
              style={{ display: 'none' }}
            />
          </label>
          {error && (
            <div style={{ marginTop: 8, color: '#ff8a80', fontSize: 12 }}>{error}</div>
          )}
        </div>

        <div style={{ padding: '8px 12px', fontSize: 11, color: '#888', borderBottom: '1px solid #333' }}>
          Legend
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 4 }}>
            {(Object.keys(NODE_COLORS) as NodeType[]).map((t) => (
              <span key={t} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                <span
                  style={{
                    width: 10,
                    height: 10,
                    borderRadius: 2,
                    background: NODE_COLORS[t],
                    display: 'inline-block',
                  }}
                />
                {t}
              </span>
            ))}
          </div>
          <div style={{ marginTop: 6, color: '#777' }}>
            Keys: arrows cycle channels, 0-9 jump, Esc clears.
          </div>
        </div>

        <div style={{ flex: 1, overflow: 'auto' }}>
          {channels.map((ch) => {
            const isSel = ch.id === selectedId;
            const pathEdges = ch.tree_edges.reduce((a, te) => a + Math.max(0, te.path.length - 1), 0);
            return (
              <div
                key={ch.id}
                onClick={() => setSelectedId(ch.id)}
                style={{
                  padding: '8px 12px',
                  borderBottom: '1px solid #2c2c2c',
                  cursor: 'pointer',
                  background: isSel ? '#37474f' : 'transparent',
                  fontSize: 12,
                }}
              >
                <div style={{ fontWeight: 600 }}>
                  Channel #{ch.id}{' '}
                  <span style={{ color: '#9ccc65', fontWeight: 400 }}>w={ch.weight.toFixed(3)}</span>
                </div>
                <div style={{ color: '#bbb' }}>root: {ch.root}</div>
                <div style={{ color: '#888' }}>
                  {ch.tree_edges.length} tree edges / {pathEdges} physical hops
                </div>
              </div>
            );
          })}
          {!channels.length && dump && (
            <div style={{ padding: 12, color: '#888', fontSize: 12 }}>No channels in dump.</div>
          )}
        </div>

        {selectedChannel && (
          <div style={{ padding: 12, borderTop: '1px solid #333', fontSize: 12 }}>
            <div style={{ fontWeight: 600 }}>Selected: Channel #{selectedChannel.id}</div>
            <div style={{ color: '#bbb' }}>weight: {selectedChannel.weight.toFixed(3)}</div>
            <div style={{ color: '#bbb' }}>tree edges: {selectedChannel.tree_edges.length}</div>
            <div style={{ color: '#bbb' }}>physical hops: {totalPathEdges}</div>
          </div>
        )}
      </aside>

      <main style={{ flex: 1, position: 'relative', background: '#1e1e1e' }}>
        {dump ? (
          <Topology dump={dump} selectedChannel={selectedChannel} />
        ) : (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexDirection: 'column',
              color: '#888',
              border: dragHover ? '2px dashed #4caf50' : '2px dashed transparent',
              transition: 'border-color 0.15s',
              margin: 16,
              borderRadius: 8,
            }}
          >
            <div style={{ fontSize: 18 }}>Drop a Blink JSON dump here</div>
            <div style={{ marginTop: 8, fontSize: 12 }}>
              or use "Open JSON..." in the sidebar.
            </div>
          </div>
        )}
        {dragHover && dump && (
          <div
            style={{
              position: 'absolute',
              inset: 16,
              border: '2px dashed #4caf50',
              borderRadius: 8,
              pointerEvents: 'none',
            }}
          />
        )}
      </main>
    </div>
  );
}
