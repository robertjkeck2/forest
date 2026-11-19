import {
  BufferAttribute, BufferGeometry, Color, Mesh, MeshStandardNodeMaterial,
  MeshBasicNodeMaterial, PlaneGeometry, RepeatWrapping, SRGBColorSpace, Texture,
} from 'three/webgpu'
import {
  attribute, cameraPosition, clamp, color, fwidth, mix, mx_noise_float, normalMap,
  normalWorld, normalWorldGeometry, positionWorld, reflector, screenUV, smoothstep, texture, uniform, uv, vec2, vec3,
} from 'three/tsl'

export const waterLevel = -0.25

export function creekCenter(z: number) {
  return 1.4 + 2.65 * Math.sin(z * .114 + .34) + .82 * Math.sin(z * .27 + 1.3)
}

export function creekWidth(z: number) {
  return 1.52 + .4 * Math.sin(z * .37) + .26 * Math.cos(z * .65 + .3)
}

export function pathCenter(z: number) {
  return creekCenter(z) - 3.4 - .45 * Math.sin(z * .16 + .7)
}

export function pathWidth(z: number) {
  return .75 + .16 * Math.cos(z * .31)
}

const ease = (x: number) => {
  const t = Math.max(0, Math.min(1, x))
  return t * t * (3 - 2 * t)
}

export function groundHeight(x: number, z: number) {
  const across = (x - creekCenter(z)) / creekWidth(z)
  const distance = Math.abs(across)
  const small = Math.sin(x * 3.7 + z * 2.8) * Math.cos(z * 3.9 - x) * .043
  const middle = .18 * Math.sin(x * .91 - z * .56) * Math.cos(z * .48)
  const large = .28 * Math.sin(x * .26 + z * .18) + .20 * Math.cos(x * .53 + z * .31)
  const shore = ease((distance - .65) / .72)
  const path = ease((Math.abs(x - pathCenter(z)) - .4) / .95)
  const unevenness = (middle + large) * shore * (.29 + .71 * path)
  const hills = ease((-z - 39) / 34) * 1.65 + ease((Math.abs(x) - 24) / 18) * 2.4
  const hillside = hills * ease((distance - 1.1) / 3.6)
  // A gravel bar on the inside of the bend keeps the transition to the water low.
  const innerBank = .36 * (1 - ease(Math.abs(across + 1.42) / .72))
    * (1 - ease(Math.abs(z + .7) / 4.3))
  return -.66 + shore * .87 + unevenness + small * (.3 + .7 * path) + hillside - innerBank
}

export function surfaceTexture(texture: Texture, srgb = false) {
  texture.wrapS = texture.wrapT = RepeatWrapping
  texture.anisotropy = 8
  if (srgb) texture.colorSpace = SRGBColorSpace
  return texture
}

type GroundTextures = {
  color: Texture
  normal: Texture
  arm: Texture
}

export function makeGround(forest: GroundTextures, gravel: GroundTextures, shadowNormalBias: number) {
  const geometry = new PlaneGeometry(82, 111, 270, 370).rotateX(-Math.PI / 2).translate(0, 0, -31)
  const positions = geometry.getAttribute('position')
  const texcoords = geometry.getAttribute('uv')
  const colors = new Float32Array(positions.count * 3)
  const groundMasks = new Float32Array(positions.count * 2)
  const floor = new Color('#a4a287')
  const trail = new Color('#beb3a3')
  const bed = new Color('#7d7463')
  const damp = new Color('#6d7951')
  const tint = new Color()

  for (let i = 0; i < positions.count; i++) {
    const x = positions.getX(i)
    const z = positions.getZ(i)
    const y = groundHeight(x, z)
    const alongPath = 1 - ease((Math.abs(x - pathCenter(z)) - pathWidth(z) * .67) * 2.5)
    const moss = Math.max(0, Math.sin(x * 1.9 + z) * Math.sin(z * .98 - x * .47)) * .66
    tint.copy(floor).lerp(damp, moss).lerp(trail, alongPath * .85)
    tint.lerp(bed, 1 - ease((y - waterLevel) * 3 + .4))
    tint.toArray(colors, i * 3)
    positions.setY(i, y)
    texcoords.setXY(i, x * .5, -z * .5)
    const bank = 1 - ease((Math.abs(x - creekCenter(z)) / creekWidth(z) - .86) * 2.8)
    groundMasks[i * 2] = Math.max(alongPath * .84, bank * .91)
    groundMasks[i * 2 + 1] = alongPath
  }
  geometry.setAttribute('color', new BufferAttribute(colors, 3))
  geometry.setAttribute('groundMasks', new BufferAttribute(groundMasks, 2))
  geometry.computeVertexNormals()
  const material = new MeshStandardNodeMaterial({ vertexColors: true })
  // Keep the receiver offset aligned with the terrain rather than its fine normal-map details.
  material.receivedShadowPositionNode = positionWorld.add(normalWorldGeometry.sub(normalWorld).mul(shadowNormalBias))
  const forestUV = uv()
  const gravelUV = forestUV.mul(1.45)
  const masks = attribute('groundMasks', 'vec2')
  const mask = masks.x
  const edgeNoise = mx_noise_float(forestUV.mul(7), .85)
  const gravelBlend = clamp(mask.add(mask.mul(mask.oneMinus()).mul(edgeNoise)), 0, 1)

  const gravelColor = surfaceTexture(gravel.color, true)
  const gravelNormal = surfaceTexture(gravel.normal)
  const baseColor = mix(texture(surfaceTexture(forest.color, true), forestUV).rgb,
    texture(gravelColor, gravelUV).rgb, gravelBlend)
  const surfaceNormal = mix(texture(surfaceTexture(forest.normal), forestUV).rgb,
    texture(gravelNormal, gravelUV).rgb, gravelBlend)

  // Mix the same gravel scan at a finer scale on the trail itself.
  const gritUV = gravelUV.mul(3.2).add(vec2(.37, .68))
  const fineGrit = texture(gravelColor, gritUV).g
  const grain = mix(1, mix(.91, 1.12, smoothstep(.045, .21, fineGrit)), masks.y)
  material.colorNode = baseColor.mul(grain)
  const gritNormal = texture(gravelNormal, gritUV).xy.sub(.5).mul(masks.y.mul(.21))
  const relief = masks.y.mul(.18).add(.91)
  material.normalNode = normalMap(vec3(surfaceNormal.xy.add(gritNormal), surfaceNormal.z), vec2(relief))

  const surface = mix(texture(surfaceTexture(forest.arm), forestUV).rg,
    texture(surfaceTexture(gravel.arm), gravelUV).rg, gravelBlend)
  material.roughnessNode = mix(.64, .98, surface.g)
  material.aoNode = mix(1, surface.r, .44)
  const ground = new Mesh(geometry, material)
  ground.receiveShadow = true
  return ground
}

export const elapsed = uniform(0)
export const motionAmount = uniform(1)

export function makeCreek() {
  const position: number[] = []
  const texcoord: number[] = []
  const indices: number[] = []
  for (let row = 0; row <= 280; row++) {
    const z = 23 - row * .38
    for (let column = 0; column <= 8; column++) {
      const u = column / 8
      position.push(creekCenter(z) + creekWidth(z) * (u * 2 - 1) * 1.36, -z, 0)
      texcoord.push(u, row / 280)
      if (row > 0 && column > 0) {
        const a = row * 9 + column
        indices.push(a - 10, a - 9, a, a - 10, a, a - 1)
      }
    }
  }
  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(position), 3))
  geometry.setAttribute('uv', new BufferAttribute(new Float32Array(texcoord), 2))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()

  const material = new MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
  })
  const reflection = reflector({
    resolutionScale: .58, bounces: false, generateMipmaps: true, samples: 4,
  })
  const x = positionWorld.x
  const z = positionWorld.z
  const t = elapsed.mul(motionAmount)
  const wideWave = z.mul(14).add(x.mul(3.5)).sub(t.mul(3.15))
    .add(x.mul(6.1).add(t.mul(.42)).sin().mul(.73))
  const crossWave = z.mul(23).sub(x.mul(8)).sub(t.mul(5.1)).add(z.mul(4).sin().mul(.56))
  // Fade waves before their pattern becomes too fine for the camera to resolve.
  const wideWeight = smoothstep(.38, 1.38, fwidth(wideWave)).oneMinus()
  const crossWeight = smoothstep(.38, 1.38, fwidth(crossWave)).oneMinus()
  const ripples = wideWave.sin().mul(wideWeight).mul(.00125).add(crossWave.sin().mul(crossWeight).mul(.0004))
  const drift = wideWave.cos().mul(wideWeight).mul(.00057).add(crossWave.cos().mul(crossWeight).mul(.00062))
  const viewY = cameraPosition.sub(positionWorld).normalize().y.abs()
  const reflectionLod = mix(2.7, 1.65, smoothstep(.12, .62, viewY))
  const reflected = reflection.sample(screenUV.flipX().add(vec2(drift, ripples))).level(reflectionLod).rgb
  const brightness = reflected.dot(vec3(.2126, .7152, .0722))
  const mirrored = reflected.div(brightness.sub(.52).max(0).mul(.52).add(1))
  const fresnel = viewY.oneMinus().pow(4)
  const center = uv().x.sub(.5).abs().mul(2)
  const shallow = smoothstep(.34, .76, center)
  // Advect broad, broken wavelets downstream independently of the reflection.
  const flow = z.negate().add(t.mul(.28))
  const across = uv().x
  const ripplePhase = flow.mul(8).add(across.mul(18).sin().mul(.65))
  const breaks = flow.mul(2.9).sub(across.mul(8))
  const wavelets = ripplePhase.sin().mul(breaks.cos())
  const distanceFade = smoothstep(.38, 1.38, fwidth(ripplePhase).max(fwidth(breaks))).oneMinus()
  const current = wavelets.mul(distanceFade).mul(shallow.mul(.7).oneMinus())
  const body = mix(color('#354b43'), color('#5b6252'), shallow)
  const reflectionAmount = clamp(fresnel.mul(.42).add(.10).sub(shallow.mul(.075)), .06, .52)
  material.colorNode = mix(body, mirrored, reflectionAmount)
    .add(color('#b2c6b5').mul(current.mul(.024)))
  material.opacityNode = clamp(fresnel.mul(.47).add(.43).sub(shallow.mul(.19)), .25, .88)

  const water = new Mesh(geometry, material)
  water.rotation.x = -Math.PI / 2
  water.position.y = waterLevel
  water.add(reflection.target)
  return water
}
