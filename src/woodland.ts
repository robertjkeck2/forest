import {
  BufferAttribute, BufferGeometry, Color, DoubleSide, Group, IcosahedronGeometry,
  InstancedBufferAttribute, InstancedMesh, MathUtils, Mesh, MeshStandardMaterial, MeshStandardNodeMaterial,
  Object3D, Raycaster, Vector2, Vector3,
  SRGBColorSpace, type Texture,
} from 'three/webgpu'
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js'
import {
  attribute, cameraPosition, color, materialColor, materialRoughness, mix, mx_noise_float,
  normalWorld, normalWorldGeometry, positionGeometry, positionLocal, positionWorld, smoothstep, vec3,
} from 'three/tsl'
import {
  creekCenter, creekWidth, elapsed, groundHeight, motionAmount, pathCenter, pathWidth,
  surfaceTexture, waterLevel,
} from './landscape'

function randomSource(seed: number) {
  return () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
    return seed / 4294967296
  }
}

// Keep the understory layout independent from changes to tree shapes.
const random = randomSource(1198035909)
const treeRandom = randomSource(291)
const treeRange = (min: number, max: number) => min + treeRandom() * (max - min)
const range = (min: number, max: number) => min + random() * (max - min)
const up = new Vector3(0, 1, 0)

type BarkHeight = (u: number, v: number) => number

export function readBarkHeight(image: HTMLImageElement): BarkHeight {
  const canvas = document.createElement('canvas')
  canvas.width = image.naturalWidth
  canvas.height = image.naturalHeight
  const { width, height } = canvas
  const context = canvas.getContext('2d')!
  context.drawImage(image, 0, 0)
  const pixels = context.getImageData(0, 0, width, height).data

  return (u, v) => {
    const x = MathUtils.euclideanModulo(u, 1) * width
    const y = MathUtils.euclideanModulo(1 - v, 1) * height
    const left = Math.floor(x)
    const top = Math.floor(y)
    const right = (left + 1) % width
    const bottom = (top + 1) % height
    const a = MathUtils.lerp(pixels[(top * width + left) * 4], pixels[(top * width + right) * 4], x - left)
    const b = MathUtils.lerp(pixels[(bottom * width + left) * 4], pixels[(bottom * width + right) * 4], x - left)
    return MathUtils.clamp(MathUtils.lerp(a, b, y - top) / 255 - .25, -.17, .25)
  }
}

function stemProfile(sides: number) {
  return {
    radii: Array.from({ length: sides }, () => treeRange(.88, 1.12)),
    roots: Array.from({ length: sides }, () => treeRange(.84, 1.24)),
  }
}

function woodyStem(points: Vector3[], widths: number[], sides: number) {
  return stemGeometry(points, widths, sides, stemProfile(sides))
}

function stemGeometry(
  points: Vector3[], widths: number[], sides: number, profile: ReturnType<typeof stemProfile>,
  relief?: { sample: BarkHeight; step: number; depth: number },
) {
  const positions: number[] = []
  const texcoords: number[] = []
  const indices: number[] = []
  const axis = points.at(-1)!.clone().sub(points[0]).normalize()
  const right = new Vector3().crossVectors(axis, Math.abs(axis.y) > .9 ? new Vector3(1, 0, 0) : up).normalize()
  const forward = new Vector3().crossVectors(axis, right).normalize()
  const barkRepeats = Math.max(1, Math.round(widths[0] * 5.15))
  const profileSize = profile.radii.length
  let distance = 0
  let row = 0

  function addRing(center: Vector3, width: number, heightAlongStem: number) {
    const buttress = Math.max(0, 1 - heightAlongStem / 1.4) ** 2
    const v = heightAlongStem * .82
    const depth = relief ? Math.min(relief.depth, width * .44) : 0
    for (let column = 0; column <= sides; column++) {
      const fraction = (column % sides) / sides
      const angle = fraction * Math.PI * 2
      const edge = fraction * profileSize
      const i = Math.floor(edge)
      const next = (i + 1) % profileSize
      const radius = MathUtils.lerp(profile.radii[i], profile.radii[next], edge - i)
      const root = MathUtils.lerp(profile.roots[i], profile.roots[next], edge - i)
      const grain = 1 + .045 * Math.sin(angle * 3 + heightAlongStem * .61)
      const u = column / sides * barkRepeats
      const ridge = relief ? relief.sample(u, v) * depth : 0
      const w = width * radius * grain * (1 + (root - 1) * buttress) + ridge
      const point = center.clone()
        .addScaledVector(right, Math.cos(angle) * w)
        .addScaledVector(forward, Math.sin(angle) * w)
      positions.push(point.x, point.y, point.z)
      texcoords.push(u, v)
      if (row && column) {
        const vertex = row * (sides + 1) + column
        indices.push(vertex - sides - 2, vertex - sides - 1, vertex, vertex - sides - 2, vertex, vertex - 1)
      }
    }
    row++
  }

  addRing(points[0], widths[0], distance)
  for (let i = 1; i < points.length; i++) {
    const span = points[i].distanceTo(points[i - 1])
    const segments = relief ? Math.ceil(span / relief.step) : 1
    for (let j = 1; j <= segments; j++) {
      const fraction = j / segments
      addRing(points[i - 1].clone().lerp(points[i], fraction),
        MathUtils.lerp(widths[i - 1], widths[i], fraction), distance + span * fraction)
    }
    distance += span
  }

  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3))
  geometry.setAttribute('uv', new BufferAttribute(new Float32Array(texcoords), 2))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  const normal = geometry.getAttribute('normal')
  for (let i = 0; i < row; i++) {
    const first = i * (sides + 1)
    const last = first + sides
    const shared = new Vector3().fromBufferAttribute(normal, first)
      .add(new Vector3().fromBufferAttribute(normal, last)).normalize()
    normal.setXYZ(first, shared.x, shared.y, shared.z)
    normal.setXYZ(last, shared.x, shared.y, shared.z)
  }
  return geometry
}

export function weatherLog(log: Object3D, barkHeight: BarkHeight) {
  log.traverse(object => {
    if (!(object instanceof Mesh) || !(object.material instanceof MeshStandardMaterial)) return
    const geometry = object.geometry as BufferGeometry
    geometry.computeBoundingBox()
    const bounds = geometry.boundingBox!
    const center = bounds.getCenter(new Vector3())
    const position = geometry.getAttribute('position')

    for (let i = 0; i < position.count; i++) {
      const x = position.getX(i), y = position.getY(i), z = position.getZ(i)
      const dy = y - center.y, dz = z - center.z
      const radius = Math.hypot(dy, dz)
      if (radius < .001) continue
      const angle = Math.atan2(dz, dy)
      const endDistance = Math.min(x - bounds.min.x, bounds.max.x - x)
      const exposedBark = MathUtils.smoothstep(endDistance, .03, .27) * MathUtils.smoothstep(radius, .025, .075)
      const grain = barkHeight(angle / Math.PI, x * 1.42)
      const chips = barkHeight(angle / Math.PI * 2, x * 2.2 + .35)
      const relief = (grain * .066 + chips * .028) * exposedBark
      position.setXYZ(i, x, y + dy / radius * relief, z + dz / radius * relief)
    }
    geometry.computeVertexNormals()
    geometry.computeBoundingSphere()
    object.material.normalScale.multiplyScalar(1.62)
    object.material.aoMap = object.material.roughnessMap
    object.material.aoMapIntensity = .72
    object.castShadow = object.receiveShadow = true
  })
}

class GrassBlades {
  private positions: number[] = []
  private colors: number[] = []
  private flex: number[] = []

  blade(base: Vector3, axis: Vector3, side: Vector3, width: number, tint: Color) {
    const center = base.clone().add(axis)
    const tip = center.clone().add(axis)
    const left = center.clone().addScaledVector(side, width)
    const right = center.clone().addScaledVector(side, -width)
    const crease = center.clone().add(new Vector3(0, width * .18, 0))
    for (const point of [base, left, crease, left, tip, crease, tip, right, crease, right, base, crease]) {
      this.positions.push(point.x, point.y, point.z)
      this.colors.push(tint.r, tint.g, tint.b)
      // Keep the root fixed and concentrate the bend toward the tip.
      this.flex.push(axis.y * (point === base ? 0 : point === tip ? 1 : .25))
    }
  }

  geometry() {
    const geometry = new BufferGeometry()
    geometry.setAttribute('position', new BufferAttribute(new Float32Array(this.positions), 3))
    geometry.setAttribute('color', new BufferAttribute(new Float32Array(this.colors), 3))
    geometry.setAttribute('bladeFlex', new BufferAttribute(new Float32Array(this.flex), 1))
    geometry.computeVertexNormals()
    return geometry
  }
}

// Pixel bounds and three cross-sections traced from each leaf in the scanned atlas.
const leafAtlas = [
  { bounds: [9, 12, 154, 433], tips: [93, 91], sections: [[34, 131], [17, 147], [46, 138]] },
  { bounds: [164, 13, 351, 408], tips: [280, 281], sections: [[192, 331], [172, 342], [212, 336]] },
  { bounds: [361, 23, 501, 375], tips: [423, 413], sections: [[391, 475], [375, 493], [378, 480]] },
  { bounds: [524, 45, 671, 377], tips: [609, 599], sections: [[556, 637], [530, 660], [538, 661]] },
  { bounds: [705, 14, 842, 429], tips: [770, 728], sections: [[732, 808], [714, 834], [719, 811]] },
  { bounds: [12, 638, 192, 1010], tips: [94, 87], sections: [[53, 167], [18, 181], [37, 143]] },
  { bounds: [215, 584, 371, 1023], tips: [322, 286], sections: [[255, 353], [225, 360], [230, 342]] },
  { bounds: [425, 611, 581, 1023], tips: [505, 512], sections: [[475, 542], [435, 571], [472, 555]] },
]
const leafStations = [.28, .53, .74]
const canopyTints = ['#e1eacb', '#eff4e6', '#bacaa2', '#faf3ce', '#e5ebd9'].map(value => new Color(value))

class Canopy {
  private positions: number[] = []
  private texcoords: number[] = []
  private colors: number[] = []
  private weights: number[] = []
  private nearIndices: number[] = []
  private farIndices: number[] = []
  private leafCount = 0

  constructor(private detailed: boolean) {}

  leaf(base: Vector3, direction: Vector3, across: Vector3, length: number) {
    const scan = leafAtlas[Math.floor(treeRandom() * leafAtlas.length)]
    const [left, top, right, bottom] = scan.bounds
    const [rootU, tipU] = scan.tips
    const halfWidth = length * (right - left) / (bottom - top) * treeRange(.79, 1.02)
    const normal = new Vector3().crossVectors(direction, across)
    const curl = halfWidth * treeRange(-.43, .52)
    const tint = canopyTints[Math.floor(treeRandom() * canopyTints.length)]
    const point = new Vector3()
    const vertex = (t: number, u: number, lift: number, shade = 1) => {
      const index = this.positions.length / 3
      point.copy(base).addScaledVector(direction, length * t)
        .addScaledVector(across, (u - rootU) * 2 * halfWidth / (right - left))
        .addScaledVector(normal, lift + curl * t * t)
      this.positions.push(point.x, point.y, point.z)
      this.texcoords.push(u / 1024, 1 - (bottom - (bottom - top) * t) / 1024)
      this.colors.push(tint.r * shade, tint.g * shade, tint.b * shade)
      this.weights.push(.35 + t * .65)
      return index
    }

    const root = vertex(.05, rootU, 0)
    const rings = (this.detailed ? [0, 1, 2] : [1]).map(station => {
      const t = leafStations[station]
      const [edgeA, edgeB] = scan.sections[station]
      const midrib = (edgeA + edgeB) * .5
      const lift = halfWidth * Math.sin(t * Math.PI)
      return [
        vertex(t, edgeA + 1, lift * .25),
        vertex(t, midrib, lift * .105),
        vertex(t, edgeB - 1, lift * .21),
        vertex(t, midrib, lift * -.075, .83),
      ]
    })
    const tip = vertex(.96, tipU, 0)

    for (let side = 0; side < 4; side++) {
      const next = (side + 1) % 4
      this.nearIndices.push(root, rings[0][side], rings[0][next])
      for (let row = 1; row < rings.length; row++) {
        const before = rings[row - 1]
        const after = rings[row]
        this.nearIndices.push(before[side], after[side], before[next], before[next], after[side], after[next])
      }
      const last = rings[rings.length - 1]
      this.nearIndices.push(last[side], tip, last[next])
    }

    if (this.leafCount++ % 5 !== 0) {
      // A four-face solid keeps a readable folded profile in the distant crowns.
      const [rimA, , rimB] = rings[this.detailed ? 1 : 0]
      this.farIndices.push(root, rimA, rimB, rimA, tip, rimB, root, tip, rimA, root, rimB, tip)
    }
  }

  geometry() {
    const attributes = {
      position: new BufferAttribute(new Float32Array(this.positions), 3),
      uv: new BufferAttribute(new Float32Array(this.texcoords), 2),
      color: new BufferAttribute(new Float32Array(this.colors), 3),
      flutter: new BufferAttribute(new Float32Array(this.weights), 1),
    }
    function build(indices: number[]) {
      const geometry = new BufferGeometry()
      for (const [name, value] of Object.entries(attributes)) geometry.setAttribute(name, value)
      geometry.setIndex(indices)
      geometry.computeVertexNormals()
      return geometry
    }
    return { near: build(this.nearIndices), far: build(this.farIndices) }
  }
}

function addLeafSpray(crown: Canopy, stems: BufferGeometry[], start: Vector3, end: Vector3, size: number) {
  const middle = start.clone().lerp(end, .55).addScaledVector(up, size * .16)
  stems.push(woodyStem([start, middle, end], [size * .033, size * .017, .0015], 4))
  const along = end.clone().sub(start).normalize()
  const across = new Vector3().crossVectors(up, along).normalize()
  const count = 10
  for (let i = 0; i < count; i++) {
    const t = .19 + i / (count - 1) * .78
    const base = t < .55 ? start.clone().lerp(middle, t / .55) : middle.clone().lerp(end, (t - .55) / .45)
    const sign = i % 2 === 0 ? 1 : -1
    const direction = along.clone().multiplyScalar(treeRange(.23, .87))
      .addScaledVector(across, sign).addScaledVector(up, treeRange(-.65, .4)).normalize()
    const side = new Vector3().crossVectors(up, direction).normalize()
      .applyAxisAngle(direction, treeRange(-1.15, 1.15))
    crown.leaf(base, direction, side, size * treeRange(.65, 1.3) * (1 - t * .37))
  }
}

const treeVariants = 7

function treeShape(index: number, barkHeight: BarkHeight) {
  const hero = index >= treeVariants
  const height = hero ? 12.6 : treeRange(10, 17)
  const radius = hero ? .52 : treeRange(.22, .45) * (index === 0 ? 1.4 : 1)
  const lean = treeRange(-1.75, 1.75)
  const turn = treeRange(-1.5, 1.5)
  const crook = treeRange(-1.35, 1.35)
  const stemAt = (t: number) => new Vector3(
    lean * t + Math.sin(t * 4.8) * crook * t,
    height * t,
    turn * t * t + Math.sin(t * 7) * .35 * t,
  )
  const trunkCenters = [0, .025, .06, .12, .2, .29, .38, .47, .56, .65, .74, .83, .92, 1].map(stemAt)
  const trunkWidths = [1.56, 1.16, .99, .93, .88, .81, .74, .66, .56, .46, .34, .23, .13, .033].map(w => w * radius)
  const profile = stemProfile(18)
  const nearTrunk = stemGeometry(trunkCenters, trunkWidths, 56, profile, { sample: barkHeight, step: .115, depth: .23 })
  const farTrunk = stemGeometry(trunkCenters, trunkWidths, 24, profile, { sample: barkHeight, step: .46, depth: .11 })
  const stems: BufferGeometry[] = []
  const twigs: BufferGeometry[] = []
  const crown = new Canopy(hero)
  let birdBranch: Vector3 | undefined
  for (let i = 0; i < 6; i++) {
    const angle = i * Math.PI / 3 + treeRange(-.44, .44)
    const root = new Vector3(Math.cos(angle), 0, Math.sin(angle)).multiplyScalar(radius * treeRange(1.6, 3.3))
    stems.push(woodyStem([
      new Vector3(0, radius * treeRange(.57, 1.39), 0),
      root.clone().multiplyScalar(.49).setY(-radius * .1),
      root.clone().multiplyScalar(.78).setY(-.16),
      root.setY(-.37),
    ], [radius * .47, radius * .29, radius * .14, .012], 8))
  }

  const branchCount = 19
  const phase = treeRange(0, 6)
  const crownBase = index === 0 ? .3 : treeRange(.32, .48)
  const canopyWidth = treeRange(.97, 1.35)
  for (let i = 0; i < branchCount; i++) {
    const progress = i / (branchCount - 1)
    const t = crownBase + (.95 - crownBase) * progress + treeRange(-.025, .025)
    const base = stemAt(t)
    const angle = i * 2.4 + phase + treeRange(-.48, .48)
    const span = treeRange(2.15, 3.85) * (1 - progress * .53) * canopyWidth
    const out = new Vector3(Math.cos(angle), 0, Math.sin(angle))
    const side = new Vector3(-out.z, 0, out.x)
    const sweep = treeRange(-.65, .65)
    const rise = span * treeRange(.24, .93)
    const elbow = base.clone().addScaledVector(out, span * .37).addScaledVector(up, rise * .25)
    const middle = base.clone().addScaledVector(out, span * .69).addScaledVector(side, sweep * .5).addScaledVector(up, rise * .62)
    const tip = base.clone().addScaledVector(out, span).addScaledVector(side, sweep).addScaledVector(up, rise)
    const width = radius * treeRange(.18, .39) * (1 - progress * .61)
    stems.push(woodyStem([base, elbow, middle, tip], [width, width * .71, width * .41, .006], 9))

    for (let j = 0; j < 4; j++) {
      const sign = j % 2 === 0 ? -1 : 1
      const origin = j < 2
        ? elbow.clone().lerp(middle, .22 + j * .52)
        : middle.clone().lerp(tip, .24 + (j - 2) * .49)
      const end = origin.clone().addScaledVector(out, span * treeRange(.15, .36))
        .addScaledVector(side, sign * span * treeRange(.23, .49)).addScaledVector(up, treeRange(.16, .72))
      const fork = origin.clone().lerp(end, .53).addScaledVector(up, -.09)
      stems.push(woodyStem([origin, fork, end], [width * .38, width * .21, .0035], 5))

      for (let k = 0; k < 3; k++) {
        const branchlet = fork.clone().lerp(end, k * .46)
        const length = treeRange(.32, .83)
        const destination = branchlet.clone().addScaledVector(out, length * treeRange(.45, .98))
          .addScaledVector(side, length * (k === 1 ? -sign : sign) * treeRange(.23, .76))
          .addScaledVector(up, length * treeRange(-.25, .46))
        addLeafSpray(crown, twigs, branchlet, destination, treeRange(.26, .39))
      }
    }
    addLeafSpray(crown, twigs, middle, tip, treeRange(.24, .37))
  }

  const top = stemAt(.93)
  for (let i = 0; i < 7; i++) {
    const angle = i * 2.4 + phase
    const end = top.clone().add(new Vector3(Math.cos(angle) * .63, treeRange(.45, 1.05), Math.sin(angle) * .63))
    addLeafSpray(crown, twigs, top, end, treeRange(.24, .37))
  }

  for (let i = 0; i < 5; i++) {
    const base = stemAt(.29 + i * .085)
    const angle = phase + i * 2.4 + .85
    const reach = treeRange(1.3, 2.6)
    const out = new Vector3(Math.cos(angle), 0, Math.sin(angle))
    const side = new Vector3(-out.z, 0, out.x)
    const middle = base.clone().addScaledVector(out, reach * .48).addScaledVector(up, reach * .19)
    const tip = base.clone().addScaledVector(out, reach).addScaledVector(up, reach * .36)
    stems.push(woodyStem([base, middle, tip], [radius * .19, radius * .094, .004], 7))
    for (let j = 0; j < 5; j++) {
      const start = middle.clone().lerp(tip, j / 5)
      const sign = j % 2 === 0 ? -1 : 1
      const end = start.clone().addScaledVector(out, treeRange(.29, .71))
        .addScaledVector(side, sign * treeRange(.36, .83)).addScaledVector(up, treeRange(-.22, .28))
      addLeafSpray(crown, twigs, start, end, treeRange(.27, .39))
    }
  }

  if (hero) {
    // Frame the creek with low boughs and visible individual leaves.
    const start = stemAt(index === treeVariants ? .224 : .27)
    const elbow = start.clone().add(new Vector3(1.35, .31, .08))
    const middle = start.clone().add(new Vector3(2.55, .54, .35))
    const tip = start.clone().add(new Vector3(3.7, .63, .54))
    stems.push(woodyStem([start, elbow, middle, tip], [.17, .11, .053, .005], 12))
    for (let i = 0; i < 10; i++) {
      const base = i < 5 ? elbow.clone().lerp(middle, i / 5) : middle.clone().lerp(tip, (i - 5) / 6)
      const side = i % 2 === 0 ? -1 : 1
      const end = base.clone().add(new Vector3(treeRange(.29, .94), treeRange(-.38, .23), side * treeRange(.36, 1.04)))
      const fork = base.clone().lerp(end, .48)
      addLeafSpray(crown, twigs, base, end, treeRange(.24, .37))
      addLeafSpray(crown, twigs, fork, fork.clone().add(new Vector3(.51, treeRange(-.22, .14), side * -.39)), .28)
    }
    if (index === treeVariants + 1) {
      const joint = middle.clone().lerp(tip, .48)
      const bend = joint.clone().add(new Vector3(.36, -.52, -.32))
      const end = joint.clone().add(new Vector3(.83, -.77, -.61))
      stems.push(woodyStem([joint, bend, end], [.042, .029, .008], 9))
      birdBranch = bend.clone().lerp(end, .67)
    }
  }

  const timber = {
    near: mergeGeometries([nearTrunk, ...stems, ...twigs]),
    far: mergeGeometries([farTrunk, ...stems]),
  }
  for (const geometry of [nearTrunk, farTrunk, ...stems, ...twigs]) geometry.dispose()
  return { timber, canopy: crown.geometry(), birdBranch }
}

type Placement = { x: number; y: number; z: number; scale: number; angle: number }

type TreeTextures = {
  bark: Texture
  barkNormal: Texture
  barkArm: Texture
  leaf: Texture
  leafAlpha: Texture
  leafNormal: Texture
}

function withMossAndLichen(base: MeshStandardMaterial, onBark: boolean) {
  const material = new MeshStandardNodeMaterial().copy(base)
  const coverage = attribute('growth', 'float')
  const field = positionWorld.mul(5.2)
  const grain = mx_noise_float(field.mul(11.6), .5, .5)
  const patches = mx_noise_float(field, .84).add(mx_noise_float(field.mul(3.7), .31))
  const wetness = onBark
    ? smoothstep(.08, 1.4, positionGeometry.y).oneMinus()
    : smoothstep(.02, .65, normalWorldGeometry.y)
      .mul(smoothstep(waterLevel - .09, waterLevel + .18, positionWorld.y))
  const moss = smoothstep(.05, .21, patches.sub(wetness.oneMinus().mul(.38)))
    .mul(wetness).mul(coverage)
  const lichen = smoothstep(.24, .44, mx_noise_float(field.mul(2.9).add(19), 1).add(grain.mul(.12)))
    .mul(wetness).mul(coverage).mul(moss.oneMinus()).mul(.62)

  const mossColor = mix(color('#293618'), color('#687b40'), grain)
  const lichenColor = mix(color('#778370'), color('#aab299'), grain)
  material.colorNode = mix(mix(materialColor.rgb, mossColor, moss), lichenColor, lichen)
  material.roughnessNode = mix(materialRoughness, .97, moss.max(lichen))
  return material
}

function makeBrush(bark: MeshStandardMaterial, leaves: MeshStandardNodeMaterial) {
  const brush = new Group()
  const random = randomSource(4926)
  const pick = (min: number, max: number) => min + random() * (max - min)
  const varieties = 4
  const clumps: Placement[][] = Array.from({ length: varieties }, () => [])
  const plant = (x: number, z: number, scale: number) => {
    const y = groundHeight(x, z)
    if (Math.abs(x - pathCenter(z)) < pathWidth(z) + 1.05) return
    if (Math.abs(x - creekCenter(z)) < creekWidth(z) + .98 || y < waterLevel + .28) return
    const variety = Math.floor(random() * varieties)
    clumps[variety].push({ x, y: y - .065, z, scale, angle: pick(0, 6.28) })
  }

  for (let row = -63; row <= 7; row += 3.2) {
    for (let column = -34; column <= 34; column += 3.2) {
      const habitat = Math.sin(column * .27 + row * .163) * Math.cos(row * .4 - column * .18)
      const distant = row < -17
      if (random() > (distant ? .75 : .4) + habitat * .28) continue
      plant(column + pick(-1.48, 1.48), row + pick(-1.48, 1.48), pick(.65, distant ? 1.5 : 1.12))
    }
  }

  // Layer connected thickets into the larger gaps visible from the path.
  const openings = [
    [-6.7, -9.4], [11, -4], [14.7, -8.4], [18.5, -14.8], [7.6, -11.5],
    [-16.5, -25.8], [-6, -41.5], [5.9, -48.5], [-2, -52.5], [6.3, -57.4],
  ]
  for (const [x, z] of openings) {
    for (let i = 0; i < 7; i++) {
      plant(x + pick(-1.65, 1.65), z + pick(-1.55, 1.55), pick(1.02, z < -22 ? 1.72 : 1.35))
    }
  }

  const transform = new Object3D()
  const tint = new Color()
  for (const [variety, placements] of clumps.entries()) {
    const canopy = new Canopy(true)
    const stems: BufferGeometry[] = []
    const shrubHeight = [.55, .88, .68, .44][variety]
    for (let i = 0; i < 9; i++) {
      const angle = i * 2.4 + pick(-.29, .29)
      const direction = new Vector3(Math.cos(angle), 0, Math.sin(angle))
      const across = new Vector3(-direction.z, 0, direction.x)
      const reach = pick(.4, .78)
      const height = shrubHeight * pick(.63, 1.02)
      const base = direction.clone().multiplyScalar(pick(.03, .15))
      const middle = base.clone().addScaledVector(direction, reach * .43).setY(height * .64)
      const tip = base.clone().addScaledVector(direction, reach).setY(height)
      stems.push(woodyStem([base, middle, tip], [.018, .009, .002], 5))
      for (let j = 0; j < 4; j++) {
        const side = j % 2 === 0 ? -1 : 1
        const start = middle.clone().lerp(tip, j * .28)
        const end = start.clone().addScaledVector(direction, pick(.12, .35))
          .addScaledVector(across, side * pick(.18, .41)).addScaledVector(up, pick(-.05, .19))
        addLeafSpray(canopy, stems, start, end, pick(.13, .24))
      }
    }
    const woody = mergeGeometries(stems)
    stems.forEach(geometry => geometry.dispose())
    const detail = canopy.geometry()
    const nearby = (place: Placement) => place.z > -20 && Math.abs(place.x) < 16
    const groups = { near: placements.filter(nearby), far: placements.filter(place => !nearby(place)) }
    for (const level of ['near', 'far'] as const) {
      const instances = groups[level]
      const foliage = new InstancedMesh(detail[level], leaves, instances.length)
      const branches = level === 'near' ? new InstancedMesh(woody, bark, instances.length) : null
      for (const [i, place] of instances.entries()) {
        transform.position.set(place.x, place.y, place.z)
        transform.rotation.y = place.angle
        transform.scale.set(place.scale * pick(.8, 1.21), place.scale, place.scale)
        transform.updateMatrix()
        foliage.setMatrixAt(i, transform.matrix)
        branches?.setMatrixAt(i, transform.matrix)
        tint.setHSL(pick(.16, .24), pick(.11, .3), pick(.66, .94))
        foliage.setColorAt(i, tint)
      }
      foliage.receiveShadow = true
      brush.add(foliage)
      if (branches) {
        branches.receiveShadow = true
        brush.add(branches)
      }
    }
  }
  return brush
}

export function makeWoodland(textures: TreeTextures, sunlight: Vector3, barkHeight: BarkHeight) {
  const trees = new Group()
  const birdPerch = new Vector3()
  const barkSurface = surfaceTexture(textures.barkArm)
  const bark = new MeshStandardMaterial({
    map: surfaceTexture(textures.bark, true),
    normalMap: surfaceTexture(textures.barkNormal),
    normalScale: new Vector2(1.42, 1.42),
    aoMap: barkSurface,
    aoMapIntensity: .62,
    roughnessMap: barkSurface,
    color: '#ded1bd',
  })
  const mossyBark = withMossAndLichen(bark, true)
  textures.leaf.colorSpace = SRGBColorSpace
  const leaves = new MeshStandardNodeMaterial({
    map: textures.leaf, alphaMap: textures.leafAlpha, normalMap: textures.leafNormal,
    normalScale: new Vector2(.68, .68), alphaTest: .43, alphaToCoverage: true,
    side: DoubleSide, vertexColors: true, roughness: .69,
  })
  // Thin leaves transmit more sunlight when viewed from the shaded side.
  const light = vec3(sunlight.x, sunlight.y, sunlight.z)
  const view = cameraPosition.sub(positionWorld).normalize()
  const backlight = normalWorld.dot(light).negate().max(0)
    .mul(view.dot(light).negate().mul(.55).add(.45).max(.1))
  leaves.emissiveNode = materialColor.rgb.mul(vec3(1.15, 1.28, .56)).mul(backlight.mul(.95).add(.075))

  const phase = positionLocal.x.mul(.62).add(positionLocal.z.mul(.57)).add(elapsed.mul(.88))
  const sway = phase.sin().mul(.031).mul(motionAmount)
  const flutter = phase.mul(2.3).add(positionLocal.y.mul(4.5)).sin()
    .mul(.017).mul(attribute('flutter', 'float')).mul(motionAmount)
  leaves.positionNode = positionLocal.add(vec3(sway, sway.mul(.24).add(flutter), sway.mul(.44)))
  const placements: Placement[][] = Array.from({ length: treeVariants + 2 }, () => [])
  const place = (x: number, z: number, scale: number, variant: number, fixedAngle?: number) => {
    const angle = treeRange(0, 6.28)
    placements[variant].push({ x, z, y: groundHeight(x, z) - .09, scale, angle: fixedAngle ?? angle })
  }

  place(-3.6, 2.8, 1.13, treeVariants, -.2)
  place(7.1, -2.8, 1.08, treeVariants + 1, 3.16)
  place(3.05, -8, 1.08, 2)
  place(-7.3, -4.3, 1.12, 4)
  place(-3.5, -16.2, 1.06, 1)

  for (let i = 0; i < 335; i++) {
    const x = treeRange(-37, 37)
    const z = treeRange(-68, 9)
    if (Math.abs(x - creekCenter(z)) < creekWidth(z) + .65) continue
    if (Math.abs(x - pathCenter(z)) < 1.46) continue
    if (z > -3 && x > -4.8 && x < 5.6) continue
    place(x, z, treeRange(.59, 1.27), i % treeVariants)
  }

  // Young trees add lower layers between the mature crowns and close gaps at the far edge.
  for (let i = 0; i < 56; i++) {
    const z = treeRange(-65, -14)
    const bank = i % 2 === 0 ? -1 : 1
    const x = creekCenter(z) + bank * treeRange(3, 27)
    if (Math.abs(x - pathCenter(z)) < 1.6) continue
    place(x, z, treeRange(.28, .56), i % treeVariants)
  }

  place(6.2, -12.7, .37, 2)
  place(9.6, -8.2, .35, 4)
  place(-5.4, -11.4, .35, 3)

  // Stagger the extra background trunks behind the wider gaps on either side of the creek.
  const background: Placement[][] = Array.from({ length: placements.length }, () => [])
  const backgroundRandom = randomSource(846)
  const backgroundRange = (min: number, max: number) => min + backgroundRandom() * (max - min)
  const openings = [
    [19.3, -18.1, 1.07], [23, -23.5, 1.11], [27.8, -29.4, .95],
    [26, -22.5, 1.18], [30.2, -27.3, 1.12], [36.9, -35.3, .98],
    [30.7, -21.2, .89], [37.4, -27.8, 1.09], [40.1, -27.4, .94],
    [18.4, -26, 1.1], [24.8, -35.2, 1.06], [34.1, -49.8, 1.14],
    [37.6, -54.3, .94], [40.1, -57.2, .87],
    [-24.2, -34.6, .86], [-27.9, -39.1, 1.02], [-30.4, -45.4, .9],
  ]
  const trunksInPlace = placements.flat()
  for (const [i, [x, z, scale]] of openings.entries()) {
    if (trunksInPlace.some(tree => Math.hypot(tree.x - x, tree.z - z) < 1.35)) continue
    const tree = { x, y: groundHeight(x, z) - .09, z, scale, angle: backgroundRange(0, 6.28) }
    background[(i * 3) % treeVariants].push(tree)
    trunksInPlace.push(tree)
  }

  const dummy = new Object3D()
  const tint = new Color()
  for (const [variant, positions] of placements.entries()) {
    const shape = treeShape(variant, barkHeight)
    const near = (place: Placement) => place.z > -25 && Math.abs(place.x) < 17
    const distant = positions.filter(place => !near(place))
    const groups = { near: positions.filter(near), far: [...distant, ...background[variant]] }
    for (const detail of ['near', 'far'] as const) {
      const batch = groups[detail]
      if (batch.length === 0) {
        shape.timber[detail].dispose()
        shape.canopy[detail].dispose()
        continue
      }
      const nearTrunks = detail === 'near'
      const growth = new Float32Array(batch.length)
      const trunks = new InstancedMesh(shape.timber[detail], nearTrunks ? mossyBark : bark, batch.length)
      if (nearTrunks) trunks.geometry.setAttribute('growth', new InstancedBufferAttribute(growth, 1))
      const crown = new InstancedMesh(shape.canopy[detail], leaves, batch.length)
      for (const [i, place] of batch.entries()) {
        const appearance = detail === 'far' && i >= distant.length ? backgroundRange : treeRange
        dummy.position.set(place.x, place.y, place.z)
        dummy.scale.set(place.scale * appearance(.83, 1.17), place.scale, place.scale)
        dummy.rotation.set(0, place.angle, 0)
        dummy.updateMatrix()
        trunks.setMatrixAt(i, dummy.matrix)
        crown.setMatrixAt(i, dummy.matrix)
        if (shape.birdBranch) birdPerch.copy(shape.birdBranch).applyMatrix4(dummy.matrix)
        if (nearTrunks && (variant === treeVariants || (variant * 3 + i * 5) % 11 < 4)) {
          growth[i] = .78 + .07 * ((variant + i) % 4)
        }
        tint.setHSL(appearance(.075, .11), appearance(.04, .12), appearance(.7, .93))
        trunks.setColorAt(i, tint)
        tint.setHSL(appearance(.17, .25), appearance(.08, .21), appearance(.77, .95))
        crown.setColorAt(i, tint)
      }
      trunks.castShadow = crown.castShadow = detail === 'near'
      trunks.receiveShadow = crown.receiveShadow = true
      trees.add(trunks, crown)
      if (shape.birdBranch) {
        trunks.updateMatrixWorld(true)
        const ray = new Raycaster(birdPerch.clone().addScaledVector(up, .11), new Vector3(0, -1, 0), 0, .18)
        const surface = ray.intersectObject(trunks)[0]
        if (surface) birdPerch.copy(surface.point)
        else birdPerch.addScaledVector(up, .025)
      }
    }
  }
  trees.add(makeBrush(bark, leaves))
  return { trees, birdPerch }
}

export function makeFerns(variants: Mesh<BufferGeometry, MeshStandardMaterial>[], alpha: Texture) {
  const group = new Group()
  const first = variants[0].material
  alpha.flipY = false
  const material = new MeshStandardNodeMaterial({
    map: first.map, alphaMap: alpha, normalMap: first.normalMap,
    roughnessMap: first.roughnessMap, roughness: .92, alphaTest: .48, side: DoubleSide,
    color: '#c3cd95', emissive: '#283514', emissiveIntensity: .17,
  })
  const t = elapsed.mul(motionAmount)
  const idle = positionLocal.x.mul(.8).add(positionLocal.z.mul(.47)).sub(t.mul(.9)).sin().mul(.045)
  const bend = idle.mul(attribute('bendWeight', 'float')).mul(motionAmount)
  material.positionNode = positionLocal.add(vec3(bend, bend.mul(-.16), bend.mul(-.55)))

  const candidates: Placement[] = []
  const plant = (x: number, z: number, scale: number) => {
    const y = groundHeight(x, z)
    if (y < waterLevel + .12 || Math.abs(x - pathCenter(z)) < pathWidth(z) + .29) return
    candidates.push({ x, y, z, scale, angle: range(0, Math.PI * 2) })
  }

  for (let i = 0; i < 48; i++) {
    const z = range(-37, 11)
    const bank = random() < .5 ? -1 : 1
    const x = creekCenter(z) + bank * (creekWidth(z) + range(.3, 2.3))
    for (let j = 0; j < 7; j++) plant(x + range(-1.6, 1.6), z + range(-1.1, 1.2), range(.75, 2.15))
  }
  for (let i = 0; i < 80; i++) plant(pathCenter(8 - i * .44) - range(1.15, 3.7), 8 - i * .44, range(.74, 1.95))
  for (const [x, z, scale] of [[-1.6, 5.5, 2.3], [5.35, 3.7, 2.18], [-2.45, .7, 1.85], [3.6, -4.4, 2.25]]) {
    plant(x, z, scale)
  }

  const distant: Placement[] = []
  const fernRandom = randomSource(732)
  const away = (min: number, max: number) => min + fernRandom() * (max - min)
  for (let i = 0; i < 55; i++) {
    const z = away(-61, -15)
    const side = i % 2 === 0 ? -1 : 1
    const x = creekCenter(z) + side * away(3.15, 29)
    for (let j = 0; j < 4; j++) {
      const plantX = x + away(-1.35, 1.35)
      const plantZ = z + away(-1.4, 1.4)
      const y = groundHeight(plantX, plantZ)
      if (y < waterLevel + .24 || Math.abs(plantX - pathCenter(plantZ)) < pathWidth(plantZ) + .52) continue
      distant.push({ x: plantX, y, z: plantZ, scale: away(1.18, 2.42), angle: away(0, 6.28) })
    }
  }

  for (const [i, fern] of variants.entries()) {
    const placements = candidates.filter((_, j) => j % variants.length === i)
    const nearCount = placements.length
    placements.push(...distant.filter((_, j) => j % variants.length === i))
    const geometry = fern.geometry.clone()
    const positions = geometry.getAttribute('position')
    const weights = new Float32Array(positions.count)
    for (let j = 0; j < positions.count; j++) {
      weights[j] = Math.min(1, Math.max(0, positions.getY(j)) * 2.9) ** 1.6
    }
    geometry.setAttribute('bendWeight', new BufferAttribute(weights, 1))
    const clumps = new InstancedMesh(geometry, material, placements.length)
    const dummy = new Object3D()
    const tint = new Color()
    for (const [j, place] of placements.entries()) {
      dummy.position.set(place.x, place.y - .025, place.z)
      dummy.scale.setScalar(place.scale)
      dummy.rotation.y = place.angle
      dummy.updateMatrix()
      clumps.setMatrixAt(j, dummy.matrix)
      const tintRange = j < nearCount ? range : away
      tint.setHSL(tintRange(.12, .22), tintRange(.08, .33), tintRange(.53, .83))
      clumps.setColorAt(j, tint)
    }
    clumps.receiveShadow = true
    group.add(clumps)
  }
  return group
}

export function makeStones(map: Texture, normalMap: Texture) {
  const rockGeometry = new IcosahedronGeometry(1, 2)
  const vertices = rockGeometry.getAttribute('position')
  const texcoords = rockGeometry.getAttribute('uv')
  for (let i = 0; i < vertices.count; i++) {
    const x = vertices.getX(i), y = vertices.getY(i), z = vertices.getZ(i)
    const radius = 1 + .12 * Math.sin(8 * x + 5 * z) * Math.sin(6 * y - 2 * z)
    vertices.setXYZ(i, x * radius, y * radius, z * radius)
    texcoords.setXY(i, texcoords.getX(i) * 1.45, texcoords.getY(i))
  }
  rockGeometry.computeVertexNormals()
  const material = new MeshStandardMaterial({
    map: surfaceTexture(map, true), normalMap: surfaceTexture(normalMap), roughness: .88,
    normalScale: new Vector2(.76, .76),
  })
  const stones = new InstancedMesh(rockGeometry, material, 512)
  const dummy = new Object3D()
  const tint = new Color()
  for (let i = 0; i < stones.count; i++) {
    const z = range(-45, 11)
    const x = creekCenter(z) + range(-1.5, 1.5) * creekWidth(z)
    const size = range(.025, .15)
    dummy.position.set(x, groundHeight(x, z) - size * .06, z)
    dummy.scale.set(size * range(.85, 1.4), size * range(.37, .66), size * range(.65, 1.26))
    dummy.rotation.set(range(-.27, .27), range(0, 6.3), range(-.24, .24))
    dummy.updateMatrix()
    stones.setMatrixAt(i, dummy.matrix)
    tint.setHSL(range(.13, .24), range(.04, .22), range(.38, .68))
    stones.setColorAt(i, tint)
  }
  stones.castShadow = stones.receiveShadow = true
  return stones
}

export function makeGroundCover() {
  const greenery = new GrassBlades()
  const palette = ['#495d2b', '#626942', '#717745', '#3d4926', '#858055', '#50522d'].map(c => new Color(c))
  function addTuft(x: number, z: number, cluster: number, tintOffset: number, blades: number, pick = range) {
    for (let j = 0; j < blades; j++) {
      const a = pick(0, 6.28)
      const h = pick(.043, .21) * cluster
      const dx = Math.cos(a), dz = Math.sin(a)
      const axis = new Vector3(dx * h * .34, h * .59, dz * h * .34)
      const rootX = x + dx * h * .18
      const rootZ = z + dz * h * .18
      const root = new Vector3(rootX, groundHeight(rootX, rootZ) - .016, rootZ)
      greenery.blade(root, axis, new Vector3(-dz, 0, dx), h * pick(.03, .085), palette[(tintOffset + j) % palette.length])
    }
  }

  for (let i = 0; i < 4400; i++) {
    const z = range(-40, 11)
    const x = creekCenter(z) + range(-11, 11)
    const y = groundHeight(x, z)
    if (y < waterLevel + .12 || Math.abs(x - pathCenter(z)) < pathWidth(z) * range(.84, 1.24)) continue
    const cluster = .7 + Math.sin(x * 1.9 + z * .31) * Math.cos(z * .56) * .39
    if (random() > cluster) continue

    addTuft(x, z, cluster, i, 7)
  }

  const grassRandom = randomSource(12904)
  const away = (min: number, max: number) => min + grassRandom() * (max - min)
  for (let i = 0; i < 190; i++) {
    const z = away(-64, 6)
    const side = i % 2 === 0 ? -1 : 1
    const x = creekCenter(z) + side * away(z > -26 ? 7 : 3.6, 32)
    const height = away(.72, 1.32)
    for (let j = 0; j < 15; j++) {
      const tuftX = x + away(-1.8, 1.8)
      const tuftZ = z + away(-1.6, 1.6)
      if (groundHeight(tuftX, tuftZ) < waterLevel + .16
          || Math.abs(tuftX - pathCenter(tuftZ)) < pathWidth(tuftZ) + .32) continue
      addTuft(tuftX, tuftZ, height, i, 5, away)
    }
  }
  const material = new MeshStandardNodeMaterial({ side: DoubleSide, vertexColors: true, roughness: .94 })
  const ripple = positionLocal.x.mul(1.1).add(positionLocal.z.mul(.92)).add(elapsed.mul(.85)).sin()
    .mul(attribute('bladeFlex', 'float')).mul(.18).mul(motionAmount)
  material.positionNode = positionLocal.add(vec3(ripple, 0, ripple.mul(.3)))
  const cover = new Mesh(greenery.geometry(), material)
  cover.receiveShadow = true
  return cover
}

export function makeMossyRocks(variants: Mesh<BufferGeometry, MeshStandardMaterial>[]) {
  const rocks = new Group()
  const materials = new Map<MeshStandardMaterial, MeshStandardNodeMaterial>()
  for (const [index, source] of variants.entries()) {
    const geometry = source.geometry.clone()
    geometry.computeBoundingBox()
    const bounds = geometry.boundingBox!
    const center = bounds.getCenter(new Vector3())
    const extent = bounds.getSize(new Vector3())
    geometry.translate(-center.x, -bounds.min.y, -center.z)
    geometry.scale(1 / extent.x, 1 / extent.x, 1 / extent.x)
    const material = materials.get(source.material) ?? withMossAndLichen(source.material, false)
    materials.set(source.material, material)
    const growth = new Float32Array(7)
    geometry.setAttribute('growth', new InstancedBufferAttribute(growth, 1))
    const batch = new InstancedMesh(geometry, material, growth.length)
    const dummy = new Object3D()
    for (let i = 0; i < batch.count; i++) {
      const z = (index === 0 && i === 0) ? 4.9 : range(-30, 9)
      const side = (index + i) % 2 === 0 ? -1 : 1
      const x = creekCenter(z) + creekWidth(z) * range(.65, 1.19) * side
      const size = (index === 0 && i === 0) ? .92 : range(.26, .95)
      dummy.position.set(x, groundHeight(x, z) - size * .22, z)
      dummy.scale.setScalar(size)
      dummy.rotation.y = range(0, 6.28)
      dummy.updateMatrix()
      batch.setMatrixAt(i, dummy.matrix)
      if ((i * 3 + index * 2) % 7 < 3) growth[i] = .82 + .05 * ((index + i) % 4)
    }
    batch.castShadow = batch.receiveShadow = true
    rocks.add(batch)
  }
  return rocks
}
