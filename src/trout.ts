import {
  BufferAttribute, BufferGeometry, CatmullRomCurve3, DoubleSide, Group, Mesh,
  MeshStandardMaterial, MeshStandardNodeMaterial, SphereGeometry, Vector3,
} from 'three/webgpu'
import {
  color, mix, modelPosition, mx_noise_float, mx_worley_noise_float,
  positionGeometry, positionLocal, smoothstep, vec3,
} from 'three/tsl'
import { creekCenter, creekWidth, elapsed, groundHeight, motionAmount, waterLevel } from './landscape'

function troutBody() {
  // The profile stores longitudinal position, body depth, and half-width.
  const profile = new CatmullRomCurve3([
    [-.435, .022, .016], [-.32, .039, .025], [-.19, .078, .049], [-.035, .115, .068],
    [.12, .111, .074], [.25, .091, .059], [.36, .068, .046], [.445, .042, .032],
    [.486, .021, .016], [.497, .002, .002],
  ].map(([x, y, z]) => new Vector3(x, y, z)))
  const positions: number[] = []
  const indices: number[] = []
  const rings = 52
  const sides = 24

  for (let ring = 0; ring <= rings; ring++) {
    const profileAtRing = profile.getPoint(ring / rings)
    const headOffset = Math.max(0, profileAtRing.x - .29) * -.14
    for (let side = 0; side < sides; side++) {
      const angle = side / sides * Math.PI * 2
      positions.push(profileAtRing.x,
        headOffset + Math.cos(angle) * profileAtRing.y,
        Math.sin(angle) * profileAtRing.z)
      if (ring > 0) {
        const current = ring * sides + side
        const next = ring * sides + (side + 1) % sides
        indices.push(current - sides, next - sides, current, current, next - sides, next)
      }
    }
  }

  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  return geometry
}

type Point = [number, number, number]

function troutFins() {
  const triangles: number[] = []
  function fin(root: Point, edge: Point[]) {
    for (let i = 1; i < edge.length; i++) triangles.push(...root, ...edge[i - 1], ...edge[i])
  }

  // The tail has a shallow fork. The small adipose fin sits behind the dorsal fin.
  fin([-.396, 0, 0], [
    [-.416, .031, 0], [-.558, .124, .008], [-.618, .14, .014],
    [-.591, .095, .016], [-.554, .034, .013], [-.542, 0, .012],
    [-.566, -.055, .014], [-.613, -.127, .016], [-.555, -.113, .008], [-.416, -.025, 0],
  ])
  fin([.082, .085, 0], [
    [.119, .105, 0], [.064, .211, .01], [.021, .211, .008], [-.075, .18, .004],
    [-.123, .136, 0], [-.167, .092, 0], [-.058, .088, 0],
  ])
  fin([-.287, .045, 0], [[-.257, .057, 0], [-.283, .093, .006], [-.31, .085, .004], [-.354, .041, 0]])
  fin([-.112, -.073, 0], [[-.079, -.089, 0], [-.143, -.151, .008],
    [-.188, -.151, .003], [-.234, -.119, 0], [-.254, -.07, 0]])

  for (const side of [-1, 1]) {
    fin([.277, -.041, .042 * side], [
      [.248, -.043, .049 * side], [.13, -.092, .147 * side],
      [.065, -.09, .146 * side], [.119, -.068, .102 * side], [.165, -.049, .054 * side],
    ])
    fin([-.012, -.085, .04 * side], [
      [-.039, -.105, .054 * side], [-.109, -.148, .101 * side],
      [-.153, -.131, .081 * side], [-.179, -.082, .036 * side],
    ])
  }

  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(triangles), 3))
  geometry.computeVertexNormals()
  return geometry
}

function troutMaterials() {
  const p = positionGeometry
  const phase = elapsed.mul(motionAmount).mul(7.3).add(p.x.mul(9)).add(modelPosition.x.mul(2))
  const wave = phase.sin().mul(p.x.negate().add(.12).max(0).pow(2)).mul(.30).mul(motionAmount)
  const swim = positionLocal.add(vec3(0, wave.mul(.07), wave))
  const body = new MeshStandardNodeMaterial({ roughness: .42, metalness: .04 })
  body.positionNode = swim

  const dapple = mx_noise_float(p.mul(vec3(96, 125, 125)), .38, .62)
  const silver = mix(color('#879f9b'), color('#c8d1be'), dapple)
  const back = smoothstep(.015, .106, p.y)
  const bodyColor = mix(silver, color('#5a7450'), back)
  const lateral = smoothstep(.011, .042, p.y.sub(.006).add(p.x.mul(.037)).abs()).oneMinus()
    .mul(smoothstep(.011, .039, p.z.abs())).mul(smoothstep(.355, .46, p.x).oneMinus())
  const pink = mix(color('#a85363'), color('#c98591'), dapple)
  const speck = smoothstep(.17, .24, mx_worley_noise_float(p.mul(vec3(36, 54, 60)))).oneMinus()
    .mul(smoothstep(-.044, .057, p.y))
  const gill = smoothstep(.004, .014, p.x.sub(.25).sub(p.y.mul(.37)).abs()).oneMinus()
    .mul(smoothstep(.03, .052, p.z.abs())).mul(.54)
  body.colorNode = mix(mix(bodyColor, pink, lateral.mul(.88)), color('#28332d'), speck.max(gill))
  body.emissiveNode = body.colorNode.mul(.09)

  const fins = new MeshStandardNodeMaterial({ side: DoubleSide, roughness: .58 })
  fins.positionNode = swim
  const finMottle = mx_noise_float(p.mul(vec3(78, 58, 78)), .5, .5)
  const finHue = mix(color('#715957'), color('#748363'), smoothstep(-.06, .09, p.y))
  fins.colorNode = mix(color('#343b32'), finHue, finMottle)
  return { body, fins }
}

export function makeTrout() {
  const school = new Group()
  const bodyGeometry = troutBody()
  const finGeometry = troutFins()
  const materials = troutMaterials()
  const eyeShape = new SphereGeometry(.018, 10, 6)
  const pupilShape = new SphereGeometry(.011, 8, 5)
  const iris = new MeshStandardMaterial({ color: '#b9ab74', roughness: .34 })
  const pupil = new MeshStandardMaterial({ color: '#171c19', roughness: .24 })

  const pools = [
    { z: 4.2, lane: -.39, size: .39, phase: .45, depth: .112, heading: -.14 },
    { z: -.5, lane: .16, size: .32, phase: 2.7, depth: .153, heading: -.39 },
    { z: -6.25, lane: .36, size: .43, phase: 1.2, depth: .108, heading: .42 },
    { z: -8.2, lane: .03, size: .34, phase: 4.4, depth: .143, heading: -.47 },
  ]
  const fish = pools.map(pool => {
    const model = new Group()
    const body = new Mesh(bodyGeometry, materials.body)
    body.castShadow = body.receiveShadow = true
    const fins = new Mesh(finGeometry, materials.fins)
    fins.receiveShadow = true
    model.add(body, fins)
    for (const side of [-1, 1]) {
      const eye = new Mesh(eyeShape, iris)
      eye.position.set(.378, .023, side * .035)
      eye.scale.z = .48
      const dark = new Mesh(pupilShape, pupil)
      dark.position.set(.381, .023, side * .043)
      dark.scale.z = .33
      model.add(eye, dark)
    }
    model.scale.setScalar(pool.size)
    school.add(model)
    return { ...pool, model }
  })

  function update(seconds: number) {
    for (const { z, lane, size, phase, depth, heading, model } of fish) {
      const t = seconds * .48 + phase
      const swimZ = z + Math.sin(t) * .58
      const across = lane + Math.sin(t * .83 + phase) * .12
      const swimX = creekCenter(swimZ) + creekWidth(swimZ) * across
      const facing = creekCenter(swimZ - .35) - creekCenter(swimZ + .35)
      const bank = Math.sin(t * .88) * .26
      const y = Math.max(groundHeight(swimX, swimZ) + size * .135 + .026,
        waterLevel - depth + Math.sin(t * 1.24) * .016)
      model.position.set(swimX, Math.min(y, waterLevel - size * .212 - .024), swimZ)
      model.rotation.set(0, Math.atan2(.7, facing) + heading + bank, Math.sin(t + phase) * .055, 'YXZ')
    }
  }

  update(0)
  return { school, update }
}
