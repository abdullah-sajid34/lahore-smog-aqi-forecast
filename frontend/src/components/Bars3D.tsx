import { useRef, useState } from 'react';
import { Canvas, useFrame, ThreeEvent } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';

interface BarData { x: number; z: number; height: number; color: string; label: string; }

interface SingleBarProps { data: BarData; maxH: number; }

function SingleBar({ data, maxH }: SingleBarProps) {
  const ref  = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  const h    = (data.height / maxH) * 4 + 0.1;
  const col  = new THREE.Color(data.color);

  useFrame(() => {
    if (!ref.current) return;
    const mesh = ref.current;
    const target = hovered ? 1.08 : 1.0;
    mesh.scale.x += (target - mesh.scale.x) * 0.12;
    mesh.scale.z += (target - mesh.scale.z) * 0.12;
  });

  return (
    <group position={[data.x * 1.6 - 8.8, h / 2, data.z * 1.6]}>
      <mesh ref={ref}
        onPointerOver={(e: ThreeEvent<PointerEvent>) => { e.stopPropagation(); setHovered(true); }}
        onPointerOut={() => setHovered(false)}>
        <boxGeometry args={[1.1, h, 1.1]} />
        <meshStandardMaterial
          color={col}
          emissive={col}
          emissiveIntensity={hovered ? 0.4 : 0.15}
          roughness={0.6}
          metalness={0.1}
        />
      </mesh>
      {hovered && (
        <Html distanceFactor={8} center style={{ pointerEvents: 'none' }}>
          <div style={{
            background: '#111A2E', border: '1px solid rgba(34,211,238,0.3)',
            borderRadius: 8, padding: '4px 8px', fontSize: 11, color: '#E2E8F0',
            whiteSpace: 'nowrap',
          }}>
            <strong>{data.label}</strong>: {data.height.toFixed(1)} µg/m³
          </div>
        </Html>
      )}
    </group>
  );
}

interface Bars3DProps { data: BarData[]; }

export function Bars3D({ data }: Bars3DProps) {
  const maxH = Math.max(...data.map(d => d.height), 1);

  // Check WebGL
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl') ?? c.getContext('experimental-webgl');
    if (!gl) return (
      <div className="flex items-center justify-center h-full text-slate-500 text-sm">
        WebGL not available
      </div>
    );
  } catch {
    return <div className="text-slate-500 text-sm p-4">WebGL not available</div>;
  }

  return (
    <Canvas
      style={{ background: 'transparent', borderRadius: 12 }}
      camera={{ position: [0, 6, 14], fov: 50 }}
      dpr={Math.min(window.devicePixelRatio, 1.5)}
      gl={{ antialias: true, powerPreference: 'default', alpha: true }}
    >
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 10, 5]} intensity={0.8} />
      <pointLight position={[-5, 8, -5]} intensity={0.4} color="#22D3EE" />

      {data.map((d, i) => (
        <SingleBar key={i} data={d} maxH={maxH} />
      ))}

      {/* Floor grid */}
      <gridHelper args={[20, 12, '#1E293B', '#1E293B']} position={[0, 0, 0]} />

      <OrbitControls
        enablePan={false}
        maxPolarAngle={Math.PI / 2.1}
        minDistance={6}
        maxDistance={20}
        autoRotate={false}
      />
    </Canvas>
  );
}
