import { useRef, useEffect, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

interface ParticlesProps { pm25: number; }

function Particles({ pm25 }: ParticlesProps) {
  const ref  = useRef<THREE.InstancedMesh>(null);
  const count = Math.min(Math.floor(pm25 * 6) + 500, 4000);
  const spd   = 0.0002 + (pm25 / 500) * 0.0008;
  const col   = new THREE.Color().lerpColors(
    new THREE.Color('#00E400'), new THREE.Color('#7E0023'), Math.min(pm25 / 400, 1)
  );
  const positions = useMemo(() => {
    const a = new Float32Array(count * 3);
    for (let i = 0; i < count * 3; i++) a[i] = (Math.random() - 0.5) * 20;
    return a;
  }, [count]);
  useEffect(() => {
    if (!ref.current) return;
    const d = new THREE.Object3D();
    for (let i = 0; i < count; i++) {
      d.position.set(positions[i*3], positions[i*3+1], positions[i*3+2]);
      d.scale.setScalar(Math.random() * 0.08 + 0.02);
      d.updateMatrix();
      ref.current.setMatrixAt(i, d.matrix);
    }
    ref.current.instanceMatrix.needsUpdate = true;
  }, [count, positions]);
  useFrame(({ clock }) => {
    if (!ref.current) return;
    const t = clock.getElapsedTime();
    const d = new THREE.Object3D();
    for (let i = 0; i < count; i++) {
      const x = positions[i*3]   + Math.sin(t * spd * (i % 7 + 1) + i) * 0.01;
      const y = positions[i*3+1] + Math.cos(t * spd * (i % 5 + 1) + i) * 0.01;
      d.position.set(x, y, positions[i*3+2]);
      d.scale.setScalar(0.05);
      d.updateMatrix();
      ref.current.setMatrixAt(i, d.matrix);
    }
    ref.current.instanceMatrix.needsUpdate = true;
  });
  return (
    <instancedMesh ref={ref} args={[undefined, undefined, count]}>
      <sphereGeometry args={[1, 4, 4]} />
      <meshBasicMaterial color={col} transparent opacity={0.55} />
    </instancedMesh>
  );
}

function CameraParallax() {
  const { camera } = useThree();
  useFrame(({ mouse }) => {
    camera.position.x += (mouse.x * 2 - camera.position.x) * 0.02;
    camera.position.y += (mouse.y * 1 - camera.position.y) * 0.02;
    (camera as THREE.PerspectiveCamera).lookAt(0, 0, 0);
  });
  return null;
}

function Fallback() {
  return <div className="absolute inset-0 rounded-3xl"
    style={{ background: 'radial-gradient(ellipse at center,rgba(139,92,246,0.15) 0%,rgba(11,18,32,0) 70%)' }} />;
}

interface SmogSceneProps { pm25: number; }

export function SmogScene({ pm25 }: SmogSceneProps) {
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduced) return <Fallback />;
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') ?? canvas.getContext('experimental-webgl');
    if (!gl) return <Fallback />;
  } catch { return <Fallback />; }
  return (
    <Canvas style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
      camera={{ position: [0, 0, 10], fov: 60 }}
      dpr={Math.min(typeof window !== 'undefined' ? window.devicePixelRatio : 1, 1.5)}
      gl={{ antialias: false, powerPreference: 'default' }}>
      <fog attach="fog" args={['#0B1220', 8, 25]} />
      <ambientLight intensity={0.3} />
      <Particles pm25={pm25} />
      <CameraParallax />
    </Canvas>
  );
}