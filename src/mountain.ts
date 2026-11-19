import {
  BufferAttribute, Color, MathUtils, Mesh, MeshBasicNodeMaterial, PlaneGeometry, type Texture, Vector3,
} from 'three/webgpu'
import { mix, positionWorld, smoothstep, texture, vec2, vec3 } from 'three/tsl'
import { surfaceTexture } from './landscape'

const ridge = [
  [-140, 0], [-112, 8], [-87, 15], [-66, 12], [-52, 22], [-31, 24],
  [-21, 32], [-15, 36], [-8, 34], [-3, 43], [2, 48], [6, 45], [11, 39],
  [19, 33], [26, 34], [32, 30],
  [48, 36], [61, 28], [77, 20], [96, 15], [120, 11], [140, 0],
]

function ridgeHeight(x: number) {
  const right = ridge.findIndex(point => point[0] > x)
  if (right < 1) return 0
  const [x0, y0] = ridge[right - 1]
  const [x1, y1] = ridge[right]
  return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

function rockNoise(x: number, z: number) {
  return Math.sin(x * .24 + z * .37) * Math.sin(z * .17 - x * .37) * 1.7
    + Math.sin(x * .63 + z * .91) * Math.sin(z * .57 - x * .24) * .64
    + Math.sin(x * 1.7 - z * 1.9) * Math.sin(z * 2.1 - x * .73) * .24
}

function carvedFace(x: number, run: number) {
  // The gullies spread apart downhill and deepen below the exposed crest.
  const fan = x / (1 + run * .014) + Math.sin(run * .072 + x * .27) * 1.65
  const channels = Math.abs(Math.cos(fan * .444 - run * .045)) ** 7
  const shoulders = Math.max(0, Math.sin(fan * .31 + run * .065)) ** 1.5
  const exposed = MathUtils.smoothstep(run, 1, 12) * (1 - MathUtils.smoothstep(run, 41, 68))
  return (shoulders * 3.4 - channels * 5.6) * exposed
}

export function makeMountain(sunlight: Vector3, rock: Texture) {
  const summitX = 7
  const geometry = new PlaneGeometry(280, 188, 280, 188)
    .rotateX(-Math.PI / 2).translate(summitX, 0, -170)
  const positions = geometry.getAttribute('position')
  for (let i = 0; i < positions.count; i++) {
    const x = positions.getX(i) - summitX
    const z = positions.getZ(i)
    const spine = -177 + Math.sin(x * .039) * 12 + Math.sin(x * .093) * 5
    const run = Math.abs(z - spine)
    const flank = Math.max(0, 1 - run / (z > spine ? 83 : 57))
    const gullies = Math.sin(x * .18 + run * .043) * Math.min(run * .088, 2.5)
    const height = (ridgeHeight(x) + rockNoise(x, z) + gullies) * flank ** 1.43
      + carvedFace(x, run)
    positions.setY(i, height * .65 - 2.4)
  }
  geometry.computeVertexNormals()

  const normals = geometry.getAttribute('normal')
  const colors = new Float32Array(positions.count * 3)
  const scree = new Color('#5d6e51')
  const shale = new Color('#4d5f65')
  const granite = new Color('#7f816c')
  const distantLight = new Color('#778e9f')
  const normal = new Vector3()
  const tint = new Color()

  for (let i = 0; i < positions.count; i++) {
    normal.fromBufferAttribute(normals, i)
    const y = positions.getY(i)
    const x = positions.getX(i)
    const z = positions.getZ(i)
    const weathering = rockNoise(x, z)
    const directLight = MathUtils.smoothstep(normal.dot(sunlight), .05, .91)
    const strata = Math.sin(y * 2.2 + x * .19 + z * .28 + weathering * .65)
    const highlights = MathUtils.clamp(.17 + directLight * .67 + strata * .13 + weathering * .054, 0, 1)
    const foothills = (1 - MathUtils.smoothstep(y + weathering * 1.15, 6, 16)) * .72
    tint.copy(shale).lerp(granite, highlights).lerp(scree, foothills)
      .lerp(distantLight, .12 + foothills * .12)
    tint.toArray(colors, i * 3)
  }

  geometry.setAttribute('color', new BufferAttribute(colors, 3))
  const material = new MeshBasicNodeMaterial({ vertexColors: true })
  const rockUV = vec2(positionWorld.x.mul(.061), positionWorld.z.mul(.053).add(positionWorld.y.mul(.045)))
  const mineralGrain = texture(surfaceTexture(rock, true), rockUV).rgb.dot(vec3(.2126, .7152, .0722))
  const bareFace = smoothstep(4, 11, positionWorld.y)
  material.colorNode = vec3(mix(1, mineralGrain.mul(1.48).add(.54), bareFace.mul(.66).add(.24)))
  return new Mesh(geometry, material)
}
