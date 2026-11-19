import {
  BufferAttribute, BufferGeometry, CatmullRomCurve3, Color, ConeGeometry, CylinderGeometry, DoubleSide, Group,
  Mesh, MeshStandardMaterial, MeshStandardNodeMaterial, Object3D, Raycaster,
  SphereGeometry, Vector3,
} from 'three/webgpu'
import { color, mix, mx_noise_float, positionGeometry, smoothstep, uniform, vec3 } from 'three/tsl'
import { groundHeight } from './landscape'

type Point = [number, number, number]
type BirdMaterial = MeshStandardMaterial | MeshStandardNodeMaterial
const up = new Vector3(0, 1, 0)
const down = new Vector3(0, -1, 0)

function ellipsoid(material: BirdMaterial, center: Point, radii: Point) {
  const geometry = new SphereGeometry(1, 24, 16)
  geometry.scale(...radii).translate(...center)
  return new Mesh(geometry, material)
}

function feather(from: Point, to: Point, halfWidth: number, shades: [Color, Color], normal = up) {
  const root = new Vector3(...from)
  const tip = new Vector3(...to)
  const side = new Vector3().subVectors(tip, root).cross(normal).normalize()
  const belly = root.clone().lerp(tip, .61)
  const left = belly.clone().addScaledVector(side, halfWidth)
  const right = belly.clone().addScaledVector(side, -halfWidth)
  const ridge = belly.clone().addScaledVector(normal, halfWidth * .18)
  const geometry = new BufferGeometry()
  const positions: number[] = []
  const colors: number[] = []
  for (const point of [root, left, ridge, left, tip, ridge, tip, right, ridge, right, root, ridge]) {
    positions.push(point.x, point.y, point.z)
    const tint = point === ridge || point === root ? shades[0] : shades[1]
    colors.push(tint.r, tint.g, tint.b)
  }
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3))
  geometry.setAttribute('color', new BufferAttribute(new Float32Array(colors), 3))
  geometry.computeVertexNormals()
  return geometry
}

function robinBody() {
  // Each ring describes the silhouette height, width, and breast-to-back depth.
  const profile = new CatmullRomCurve3([
    [.143, .012, .017], [.169, .095, .148], [.23, .148, .223], [.32, .176, .251],
    [.404, .167, .226], [.479, .143, .171], [.531, .134, .162], [.594, .142, .158],
    [.648, .115, .129], [.688, .068, .078], [.706, .002, .003],
  ].map(p => new Vector3(...p)))
  const positions: number[] = []
  const indices: number[] = []
  const rings = 56
  const sides = 40
  for (let ring = 0; ring <= rings; ring++) {
    const { x: y, y: width, z: depth } = profile.getPoint(ring / rings)
    const t = Math.max(0, Math.min(1, (y - .26) / .37))
    const center = -.047 + t * t * (3 - 2 * t) * .209
    for (let side = 0; side < sides; side++) {
      const angle = side / sides * Math.PI * 2
      const back = Math.cos(angle)
      positions.push(width * Math.sin(angle), y, center + depth * back)
      if (ring > 0) {
        const v = ring * sides + side
        const n = ring * sides + (side + 1) % sides
        indices.push(v - sides, n - sides, v, v, n - sides, n)
      }
    }
  }
  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  return geometry
}

const neckPivot: Point = [0, .516, .107]

function robinPlumage() {
  const p = positionGeometry
  const pitch = uniform(0)
  const yaw = uniform(0)
  const material = new MeshStandardNodeMaterial({ roughness: .86 })
  const featherGrain = mx_noise_float(p.mul(vec3(194, 88, 168)), .65, .84)
  const shade = mx_noise_float(p.mul(vec3(48, 24, 36)), .52, .56)
  const tuft = mx_noise_float(p.mul(vec3(77, 39, 57)), .26)
  const olive = mix(color('#524a36'), color('#8c8063'), shade)
  const underside = smoothstep(.24, .43, p.y).oneMinus().mul(smoothstep(-.14, .075, p.z))
  const bibEdge = p.z.add(tuft.mul(.036))
  const bibHeight = smoothstep(.307, .424, p.y)
    .mul(smoothstep(.627, .713, p.y.add(tuft.mul(.026))).oneMinus())
  const bib = smoothstep(.122, .215, bibEdge).mul(bibHeight)
  const collar = smoothstep(.06, .148, bibEdge).mul(bibHeight)
  const rust = mix(color('#aa4725'), color('#d4793e'), shade)
  const base = mix(olive, color('#cabda6'), underside.mul(.92))
  material.colorNode = mix(mix(base, color('#8b8778'), collar.mul(.62)), rust, bib).mul(featherGrain)

  // Blend the head turn into the neck without breaking the chest silhouette.
  const flex = smoothstep(.43, .575, p.y)
  const pivot = vec3(...neckPivot)
  const local = p.sub(pivot)
  const turn = yaw.mul(flex)
  const tilt = pitch.mul(flex)
  const x = local.x.mul(turn.cos()).add(local.z.mul(turn.sin()))
  const z = local.z.mul(turn.cos()).sub(local.x.mul(turn.sin()))
  material.positionNode = pivot.add(vec3(x,
    local.y.mul(tilt.cos()).sub(z.mul(tilt.sin())),
    local.y.mul(tilt.sin()).add(z.mul(tilt.cos()))))
  return { material, pitch, yaw }
}

function limb(start: Vector3, end: Vector3, radius: number, material: MeshStandardMaterial) {
  const mesh = new Mesh(new CylinderGeometry(radius * .66, radius, start.distanceTo(end), 7), material)
  mesh.position.copy(start).lerp(end, .5)
  mesh.quaternion.setFromUnitVectors(up, end.clone().sub(start).normalize())
  return mesh
}

function makeRobin(size: number) {
  const bird = new Group()
  bird.name = 'Robin'
  const plumage = robinPlumage()
  const bill = new MeshStandardMaterial({ color: '#352d24', roughness: .57 })
  const lowerBill = new MeshStandardMaterial({ color: '#5b4b38', roughness: .66 })
  const foot = new MeshStandardMaterial({ color: '#726151', roughness: .91 })
  const eye = new MeshStandardMaterial({ color: '#15130f', roughness: .21 })
  const eyelid = new MeshStandardMaterial({ color: '#745846', roughness: .86 })
  const wing = new MeshStandardMaterial({ color: '#6c5e47', roughness: .92 })
  const feathers = new MeshStandardMaterial({ vertexColors: true, side: DoubleSide, roughness: .86 })
  const flightFeathers: [Color, Color] = [new Color('#75664e'), new Color('#4a4133')]
  const coverts: [Color, Color] = [new Color('#998266'), new Color('#695b48')]
  const tail = new Group()
  tail.position.set(0, .273, -.23)
  for (let i = -2; i <= 2; i++) {
    tail.add(new Mesh(feather([i * .022, 0, 0], [i * .033, .042, -.333 + Math.abs(i) * .02],
      .033, flightFeathers), feathers))
  }

  bird.add(new Mesh(robinBody(), plumage.material), tail)
  for (const side of [-1, 1]) {
    bird.add(ellipsoid(wing, [.127 * side, .347, -.111], [.058, .105, .192]))
    const outside = new Vector3(side * .94, .32, .13).normalize()
    for (let i = 0; i < 8; i++) {
      const y = .285 + i * .021
      bird.add(new Mesh(feather([side * (.162 + i * .002), y, .028 - i * .01],
        [side * (.126 + i * .002), y - .044, -.312 + i * .023], .027, flightFeathers, outside), feathers))
    }
    for (let i = 0; i < 7; i++) {
      const y = .331 + i * .019
      bird.add(new Mesh(feather([side * .168, y, .056 - i * .007],
        [side * (.19 - i * .003), y - .038, -.133 - i * .015], .031, coverts, outside), feathers))
    }

    const legTop = new Vector3(side * .065, .216, -.046)
    const ankle = new Vector3(side * .081, .021, .001)
    bird.add(limb(legTop, ankle, .0105, foot))
    for (const spread of [-1, 0, 1]) {
      const knuckle = new Vector3(side * .081 + spread * .026, .018, .046)
      const tip = new Vector3(side * .081 + spread * .048, -.005, .081 - Math.abs(spread) * .012)
      bird.add(limb(ankle, knuckle, .007, foot), limb(knuckle, tip, .005, bill))
    }
    bird.add(limb(ankle, new Vector3(side * .095, -.006, -.054), .006, foot))
  }

  const head = new Group()
  head.position.set(...neckPivot)
  const beak = new Mesh(new ConeGeometry(.031, .12, 10).rotateX(Math.PI * .5), bill)
  beak.scale.y = .57
  beak.position.set(0, .036, .219)
  const lower = new Mesh(new ConeGeometry(.022, .098, 10).rotateX(Math.PI * .5), lowerBill)
  lower.scale.y = .51
  lower.position.set(0, .018, .218)
  head.add(beak, lower)
  for (const side of [-1, 1]) {
    head.add(ellipsoid(eyelid, [.112 * side, .088, .139], [.01, .023, .027]),
      ellipsoid(eye, [.12 * side, .089, .142], [.009, .02, .022]))
  }
  bird.add(head)
  bird.scale.setScalar(size)
  bird.traverse(part => {
    if (part instanceof Mesh) part.castShadow = part.receiveShadow = true
  })
  return { bird, head, tail, plumage }
}

function makeHawk() {
  const bird = new Group()
  bird.name = 'Soaring hawk'
  const body = new MeshStandardMaterial({ color: '#795e47', roughness: .83 })
  const head = new MeshStandardMaterial({ color: '#695444', roughness: .83 })
  const feathers = new MeshStandardMaterial({ vertexColors: true, side: DoubleSide, roughness: .81 })
  const barred: [Color, Color] = [new Color('#baa084'), new Color('#655549')]
  const primary: [Color, Color] = [new Color('#716255'), new Color('#463a34')]
  const wings: Group[] = []
  bird.add(ellipsoid(body, [0, 0, 0], [.079, .085, .233]),
    ellipsoid(head, [0, .025, .198], [.058, .058, .081]))
  for (let i = -3; i <= 3; i++) {
    bird.add(new Mesh(feather([i * .012, -.005, -.138], [i * .032, .006, -.402 + Math.abs(i) * .018],
      .031, barred), feathers))
  }
  for (const side of [-1, 1]) {
    const wing = new Group()
    wing.scale.x = side
    wing.position.x = side * .046
    wing.add(ellipsoid(body, [.197, .015, .018], [.215, .052, .115]))
    for (let i = 0; i < 7; i++) {
      const x = .045 + i * .056
      wing.add(new Mesh(feather([x, .022, .095], [x + .1, -.01, -.2 + i * .005], .048, barred), feathers))
    }
    for (let i = 0; i < 5; i++) {
      const from: Point = [.319 + i * .012, .018, .092 - i * .033]
      const tip: Point = [.65 - i * i * .009, .041 - i * .008, .087 - i * .061]
      wing.add(new Mesh(feather(from, tip, .041, primary), feathers))
    }
    wings.push(wing)
    bird.add(wing)
  }
  return { bird, wings }
}

function logPerch(log: Object3D) {
  log.updateWorldMatrix(true, true)
  const point = log.localToWorld(new Vector3(.79, .6, -.003))
  const hit = new Raycaster(point, down).intersectObject(log, true)[0]
  return hit?.point ?? point.setY(groundHeight(point.x, point.z) + .17)
}

export function makeBirds(log: Object3D, branchPerch: Vector3) {
  const group = new Group()
  const robins = [
    { ...makeRobin(.194), perch: logPerch(log), facing: 1.03 },
    { ...makeRobin(.191), perch: branchPerch, facing: -.76 },
  ]
  const hawk = makeHawk()
  group.add(...robins.map(robin => robin.bird), hawk.bird)

  function update(time: number, motion: number) {
    const t = time * motion
    for (const [i, { bird, head, tail, plumage, perch, facing }] of robins.entries()) {
      const glance = Math.sin(t * .93 + i * 1.7)
      const quickTurn = Math.tanh(2.1 * Math.sin(t * 1.7 + i * .8))
      bird.position.copy(perch)
      bird.rotation.y = facing + Math.sin(t * .37 + i) * .21 * motion
      head.rotation.y = (.26 * glance + .19 * quickTurn) * motion
      head.rotation.x = Math.sin(t * 1.27 + i * 2.2) * .16 * motion
      plumage.yaw.value = head.rotation.y
      plumage.pitch.value = head.rotation.x
      tail.rotation.x = Math.sin(t * 2.4 + i * 1.3) * .08 * motion
    }

    const angle = t * .21 + .36
    // The orbit runs along the creek so the hawk passes through the opening above it.
    const x = .3 + Math.sin(angle) * 3.1 + Math.cos(angle) * .55
    const z = -70 - Math.cos(angle) * 7.2
    const height = 4.05 - z * .266 + Math.sin(angle * 2) * .32
    hawk.bird.position.set(x, height, z)
    const eastward = 3.1 * Math.cos(angle) - .55 * Math.sin(angle)
    const southward = 7.2 * Math.sin(angle)
    hawk.bird.rotation.set(-.042 * Math.cos(angle * 2), Math.atan2(eastward, southward),
      -.24 - Math.sin(angle) * .08, 'YXZ')
    const wingbeat = Math.max(0, Math.sin(t * .76 + 1.7) - .66)
      * Math.sin(t * 14.4) * .91 * motion
    for (const [i, wing] of hawk.wings.entries()) wing.rotation.z = (i === 0 ? -1 : 1) * (.11 + wingbeat)
  }

  update(0, 1)
  return { group, update }
}
