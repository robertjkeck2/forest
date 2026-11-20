import {
  ACESFilmicToneMapping, DirectionalLight, EquirectangularReflectionMapping,
  HemisphereLight, MathUtils, Mesh, MeshStandardMaterial, PerspectiveCamera, Scene, TextureLoader,
  Vector3, WebGPURenderer, type BufferGeometry, type DepthTexture, type Node,
} from 'three/webgpu'
import { float, Fn, mix, smoothstep, texture as sampleTexture, vec2 } from 'three/tsl'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js'
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js'
import {
  creekCenter, elapsed, groundHeight, makeCreek, makeGround, motionAmount, pathCenter,
} from './landscape'
import { makeFerns, makeGroundCover, makeMossyRocks, makeStones, makeWoodland, readBarkHeight, weatherLog } from './woodland'
import { makeMountain } from './mountain'
import { makeTrout } from './trout'
import { makeBirds } from './birds'
import './style.css'

const experience = document.querySelector<HTMLElement>('#experience')!
const viewport = document.querySelector<HTMLDivElement>('#viewport')!
const status = document.querySelector<HTMLDivElement>('#status')!
const statusMessage = document.querySelector<HTMLParagraphElement>('#status-message')!
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
const assets = import.meta.env.BASE_URL + 'assets/'

async function start() {
  const renderer = new WebGPURenderer({
    antialias: true,
    forceWebGL: new URLSearchParams(location.search).has('webgl'),
  })
  renderer.shadowMap.enabled = true
  renderer.toneMapping = ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.2

  const texture = new TextureLoader()
  const model = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder)
  const hdr = new HDRLoader()
  const [
    ground, groundNormal, groundArm, mud, mudNormal, mudArm,
    bark, barkNormal, barkArm, barkHeightMap, rock, rockNormal, alpha, fern, log, sky,
    mossyRocks, leaf, leafAlpha, leafNormal, daylight,
  ] = await Promise.all([
    texture.loadAsync(assets + 'forrest_ground_01-color.jpg'),
    texture.loadAsync(assets + 'forrest_ground_01-normal.jpg'),
    texture.loadAsync(assets + 'forrest_ground_01-arm.jpg'),
    texture.loadAsync(assets + 'path-color.jpg'),
    texture.loadAsync(assets + 'path-normal.jpg'),
    texture.loadAsync(assets + 'path-arm.jpg'),
    texture.loadAsync(assets + 'bark_brown_02-color.jpg'),
    texture.loadAsync(assets + 'bark_brown_02-normal.jpg'),
    texture.loadAsync(assets + 'bark_brown_02-arm.jpg'),
    texture.loadAsync(assets + 'bark_brown_02-height.jpg'),
    texture.loadAsync(assets + 'rock_boulder_dry-color.jpg'),
    texture.loadAsync(assets + 'rock_boulder_dry-normal.jpg'),
    texture.loadAsync(assets + 'fern_02/alpha.webp'),
    model.loadAsync(assets + 'fern_02/fern_02.meshopt.gltf'),
    model.loadAsync(assets + 'dead_tree_trunk/dead_tree_trunk.meshopt.gltf'),
    hdr.loadAsync(assets + 'woodland.hdr'),
    model.loadAsync(assets + 'rock_moss_set_01/rock_moss_set_01.meshopt.gltf'),
    texture.loadAsync(assets + 'canopy/leaf-color.webp'),
    texture.loadAsync(assets + 'canopy/leaf-alpha.webp'),
    texture.loadAsync(assets + 'canopy/leaf-normal.jpg'),
    hdr.loadAsync(assets + 'daylight-sky-1k.hdr'),
    renderer.init(),
  ])

  const scene = new Scene()
  daylight.mapping = EquirectangularReflectionMapping
  scene.background = daylight
  scene.backgroundIntensity = .64
  scene.backgroundRotation.set(-.105, .28, 0)
  sky.mapping = EquirectangularReflectionMapping
  scene.environment = sky
  scene.environmentIntensity = .72
  scene.environmentRotation.y = 1.8
  scene.add(new HemisphereLight('#e0edff', '#a38b65', 1.8))
  const sun = new DirectionalLight('#fff1ce', 3.2)
  // Light the path through the crowns across the creek.
  sun.position.set(12, 22, -12)
  sun.target.position.set(1, 0, -4)
  sun.castShadow = true
  // The sun and its major casters are stationary. Keep the canopy shadow
  // stable while the leaves and small wildlife animate in the main view.
  sun.shadow.autoUpdate = false
  sun.shadow.needsUpdate = true
  const shadowResolution = window.matchMedia('(pointer: coarse)').matches ? 2048 : 3072
  sun.shadow.mapSize.set(shadowResolution, shadowResolution)
  sun.shadow.camera.left = sun.shadow.camera.bottom = -19
  sun.shadow.camera.right = sun.shadow.camera.top = 19
  sun.shadow.camera.near = .5
  sun.shadow.camera.far = 58
  sun.shadow.normalBias = .022
  sun.shadow.bias = -.00021
  sun.shadow.intensity = .93
  const filterStep = .024 / (sun.shadow.camera.right - sun.shadow.camera.left)
  const filterNode = Fn(({ depthTexture, shadowCoord }: { depthTexture: DepthTexture; shadowCoord: Node<'vec3'> }) => {
    // Fixed light-space taps keep the penumbra stable as the camera moves.
    let visibility: Node<'float'> = float(0)
    for (const y of [-1, 0, 1]) {
      for (const x of [-1, 0, 1]) {
        const weight = (2 - Math.abs(x)) * (2 - Math.abs(y))
        const tap = sampleTexture(depthTexture, shadowCoord.xy.add(vec2(x, y).mul(filterStep)))
        visibility = visibility.add(tap.compare(shadowCoord.z).x.mul(weight))
      }
    }
    const edge = shadowCoord.xy.min(shadowCoord.xy.oneMinus())
    return mix(1, visibility.mul(1 / 16), smoothstep(0, .075, edge.x.min(edge.y)))
  })
  Object.assign(sun.shadow, { filterNode })
  scene.add(sun, sun.target)

  const ferns: Mesh<BufferGeometry, MeshStandardMaterial>[] = []
  fern.scene.traverse(object => {
    if (object instanceof Mesh && object.material instanceof MeshStandardMaterial) ferns.push(object)
  })

  const sunlight = sun.position.clone().sub(sun.target.position).normalize()
  const barkHeight = readBarkHeight(barkHeightMap.image)
  const { trees: woodland, birdPerch } = makeWoodland(
    { bark, barkNormal, barkArm, leaf, leafAlpha, leafNormal }, sunlight, barkHeight,
  )
  const terrain = makeGround(
    { color: ground, normal: groundNormal, arm: groundArm },
    { color: mud, normal: mudNormal, arm: mudArm },
    sun.shadow.normalBias,
  )
  const trout = makeTrout()
  scene.add(terrain, makeCreek(), trout.school, woodland, makeMountain(sunlight, rock),
    makeStones(rock, rockNormal), makeFerns(ferns, alpha), makeGroundCover())

  const rockVariants: Mesh<BufferGeometry, MeshStandardMaterial>[] = []
  mossyRocks.scene.traverse(object => {
    if (object instanceof Mesh && object.material instanceof MeshStandardMaterial) rockVariants.push(object)
  })
  scene.add(makeMossyRocks(rockVariants))

  const fallenTree = log.scene
  const logZ = 2.75
  const logX = pathCenter(logZ) - 1.62
  fallenTree.position.set(logX, groundHeight(logX, logZ) - .035, logZ)
  fallenTree.rotation.set(.06, -1.16, -.03)
  fallenTree.scale.setScalar(1.32)
  weatherLog(fallenTree, barkHeight)
  scene.add(fallenTree)
  const birds = makeBirds(fallenTree, birdPerch)
  scene.add(birds.group)

  const camera = new PerspectiveCamera(51, 1, .16, 330)
  const focus = new Vector3()
  let scrollDistance = 1
  let targetTravel = 0
  const updateTravel = () => {
    targetTravel = MathUtils.clamp(window.scrollY / scrollDistance, 0, 1)
  }
  viewport.append(renderer.domElement)
  const resize = () => {
    const { clientWidth: width, clientHeight: height } = viewport
    scrollDistance = Math.max(1, experience.offsetHeight - height)
    updateTravel()
    camera.aspect = width / Math.max(1, height)
    camera.fov = camera.aspect < 1 ? 64 : 51
    camera.updateProjectionMatrix()
    const pixelRatio = Math.sqrt(1_650_000 / Math.max(1, width * height))
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.25, pixelRatio))
    renderer.setSize(width, height, false)
  }
  new ResizeObserver(resize).observe(viewport)
  resize()
  window.addEventListener('scroll', updateTravel, { passive: true })

  let cameraTravel = targetTravel
  let pointerX = 0
  let pointerY = 0
  let yaw = 0
  let pitch = 0
  viewport.addEventListener('pointermove', event => {
    if (event.pointerType !== 'mouse') return
    const bounds = viewport.getBoundingClientRect()
    pointerX = (event.clientX - bounds.left) / bounds.width - .5
    pointerY = (event.clientY - bounds.top) / bounds.height - .5
  })
  viewport.addEventListener('pointerleave', () => { pointerX = 0; pointerY = 0 })

  const startTime = performance.now()
  let previous = startTime
  let firstFrame = true
  renderer.setAnimationLoop(now => {
    const time = (now - startTime) / 1000
    const dt = Math.min(.05, Math.max(0, (now - previous) / 1000))
    previous = now
    elapsed.value = time
    // Keep scrolling responsive and reduce incidental movement when requested.
    motionAmount.value = reducedMotion.matches ? .22 : 1
    trout.update(time * motionAmount.value)
    birds.update(time, motionAmount.value)
    const follow = reducedMotion.matches ? 1 : 1 - Math.exp(-dt * 5.7)
    const scrollFollow = reducedMotion.matches ? 1 : 1 - Math.exp(-dt * 10.5)
    cameraTravel += (targetTravel - cameraTravel) * scrollFollow
    yaw += ((reducedMotion.matches ? 0 : pointerX) - yaw) * follow
    pitch += ((reducedMotion.matches ? 0 : pointerY) - pitch) * follow
    const z = 8.65 - cameraTravel * 8
    const x = pathCenter(z)
    camera.position.set(x, groundHeight(x, z) + 1.57, z)
    const targetZ = z - 11.3
    const creekView = creekCenter(targetZ) + .75 - .76 * (1 - cameraTravel)
    focus.set(creekView + yaw * 3.3, .79 - pitch * 1.85, targetZ)
    camera.lookAt(focus)
    try {
      renderer.render(scene, camera)
      if (firstFrame) {
        firstFrame = false
        status.hidden = true
      }
    } catch (error) {
      renderer.setAnimationLoop(null)
      showError(error)
    }
  })
}

function showError(error: unknown) {
  console.error(error)
  status.dataset.state = 'error'
  statusMessage.textContent = 'The woodland couldn’t be rendered. Try reloading in a browser with WebGL 2 or WebGPU.'
  status.hidden = false
}

start().catch(showError)
