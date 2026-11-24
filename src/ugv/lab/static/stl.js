// Binary STL parsing and the STRL-001 assembly for the Lab dashboard.
// The manifest and meshes are generated from the Gazebo reference SDF by
// tools/build_strl_assets.py, which is the source of truth for poses,
// joint axes and limits. Meshes are authored in the model frame (every SDF visual
// carries `<pose relative_to="__model__">`), so the link pose locates the joint
// pivot, not the vertices.
import * as THREE from '/static/vendor/three.module.min.js';

const BASE = '/static/models/strl001/';

// Gazebo (x forward, y left, z up) -> three.js (X right, Y up, Z back), as in pov3d.js.
const gx = (x, y, z) => new THREE.Vector3(x, z, -y);

// SDF visual materials, by link role. Flat STL normals are kept (see parseBinarySTL): the
// generated meshes are coarse (~26° mean dihedral), so welded smooth normals read as crumpled
// foil under the panorama IBL. Surface character comes from the materials instead.
const MATERIALS = {
  airframe: { color: 0xdfe3e8, roughness: 0.55, metalness: 0.1, clearcoat: 0.25, clearcoatRoughness: 0.6 },
  control: { color: 0xf2731a, roughness: 0.5, metalness: 0.05 },
  propulsion: { color: 0x272a2f, roughness: 0.4, metalness: 0.55 },
};

// Revolute limits from model.sdf; props are continuous.
const LIMITS = {
  tilt: [-0.05, 2.0],
  control: [-0.436, 0.436],
  prop: [-Infinity, Infinity],
};

function materialFor(id) {
  if (id === 'base_link') return MATERIALS.airframe;
  if (id.includes('aileron') || id.includes('ruddervator')) return MATERIALS.control;
  return MATERIALS.propulsion;
}

function limitsFor(name) {
  if (name.endsWith('_tilt_joint')) return LIMITS.tilt;
  if (name.endsWith('_prop_joint')) return LIMITS.prop;
  return LIMITS.control;
}

// `right_nacelle_link` -> `right_tilt_joint`, `right_prop_link` -> `right_prop_joint`.
function jointName(linkId) {
  if (linkId.endsWith('_nacelle_link')) return `${linkId.slice(0, -'_nacelle_link'.length)}_tilt_joint`;
  return `${linkId.replace(/_link$/, '')}_joint`;
}

/** Parse a binary STL into a non-indexed BufferGeometry in the three.js frame. */
export function parseBinarySTL(buffer) {
  if (buffer.byteLength < 84) throw new Error('truncated binary STL');
  const view = new DataView(buffer);
  const count = view.getUint32(80, true);
  if (buffer.byteLength < 84 + count * 50) throw new Error(`truncated binary STL: ${count} triangles declared`);

  const positions = new Float32Array(count * 9);
  const normals = new Float32Array(count * 9);
  let p = 0;
  for (let i = 0; i < count; i++) {
    const o = 84 + i * 50;
    const nx = view.getFloat32(o, true);
    const ny = view.getFloat32(o + 4, true);
    const nz = view.getFloat32(o + 8, true);
    for (let v = 0; v < 3; v++) {
      const b = o + 12 + v * 12;
      positions[p] = view.getFloat32(b, true);
      positions[p + 1] = view.getFloat32(b + 8, true);
      positions[p + 2] = -view.getFloat32(b + 4, true);
      normals[p] = nx;
      normals[p + 1] = nz;
      normals[p + 2] = -ny;
      p += 3;
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

async function fetchBuffer(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${url}`);
  return response.arrayBuffer();
}

/**
 * Fetch the manifest and every mesh, then assemble the articulated model.
 * Resolves to { root, joints }; `joints[name]` is { pivot, axis, min, max, angle }.
 * `onPart` fires after each mesh lands, so the caller can mark its view dirty.
 */
export async function loadStrlModel(onPart, makeCard) {
  const response = await fetch(BASE + 'manifest.json');
  if (!response.ok) throw new Error(`${response.status} manifest.json`);
  const manifest = await response.json();

  const root = new THREE.Group();
  const nodes = new Map();
  const joints = {};
  const absPose = new Map(manifest.parts.map((part) => [part.id, part.pose]));

  // Pivots first: a child may finish loading before its parent. Manifest poses are
  // absolute model-frame, so a child pivot carries only its offset from the parent.
  for (const part of manifest.parts) {
    const [px, py, pz, roll, pitch, yaw] = part.pose;
    const parent = part.joint?.parent ? absPose.get(part.joint.parent) : null;
    const pivot = new THREE.Group();
    pivot.position.copy(gx(px - (parent?.[0] ?? 0), py - (parent?.[1] ?? 0), pz - (parent?.[2] ?? 0)));
    nodes.set(part.id, pivot);
    if (part.joint && part.joint.type === 'revolute') {
      const name = jointName(part.id);
      const [min, max] = limitsFor(name);
      const axis = new THREE.Vector3(...part.joint.axis).applyEuler(new THREE.Euler(roll, pitch, yaw));
      joints[name] = { pivot, axis: gx(axis.x, axis.y, axis.z).normalize(), min, max, angle: 0 };
    }
  }
  for (const part of manifest.parts) {
    const parent = part.joint?.parent;
    (parent && nodes.get(parent) ? nodes.get(parent) : root).add(nodes.get(part.id));
  }

  // ArUco cards ride on the model frame (positions and outward normals from tags.json).
  if (makeCard) {
    for (const tag of manifest.tags || []) {
      const card = makeCard(tag.id);
      card.position.copy(gx(...tag.pos));
      card.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), gx(...tag.normal).normalize());
      root.add(card);
    }
  }

  // Small parts first so something appears while the 8 MB body is still in flight.
  const parts = manifest.parts.filter((part) => part.file).sort((a, b) => a.tris - b.tris);
  for (const part of parts) {
    const [px, py, pz] = part.pose;
    const geometry = parseBinarySTL(await fetchBuffer(BASE + part.file));
    const mesh = new THREE.Mesh(geometry, new THREE.MeshPhysicalMaterial(materialFor(part.id)));
    // Undo the pivot offset: the vertices are already in the model frame.
    mesh.position.copy(gx(-px, -py, -pz));
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    nodes.get(part.id).add(mesh);
    onPart?.();
  }

  // Gear on the floor, and the SDF nose (+x) turned to the policy heading (-y, the
  // direction the placeholder pointed). Dimensions are untouched.
  const box = new THREE.Box3().setFromObject(root);
  root.position.y = -box.min.y;
  root.rotation.y = -Math.PI / 2;
  return { root, joints };
}
