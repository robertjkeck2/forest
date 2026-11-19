# Where the Water Turns

An interactive woodland creek study built with Three.js, TypeScript, and Vite. The view follows a short path along a shallow creek through a sunlit forest, with a mountain beyond the creek bend.

```sh
npm ci
npm run dev
```

Scroll down to follow the path and up to go back. Mouse movement gently shifts the gaze. The scene respects the browser's reduced-motion setting.

The terrain, camera path, and creek share the same curves in `src/landscape.ts`. The creek uses a planar reflection with animated TSL ripples and a shallow edge that shows the bed. `src/woodland.ts` uses instanced meshes for trees, stones, and scanned fern clumps. Ferns bend in the vertex shader with stronger movement at the frond tips. The canopy places scanned leaves along branching twigs and shades them with thin-leaf backlighting.

`npm run build` checks TypeScript and builds the static site into `dist/`. The renderer uses WebGPU with a WebGL 2 fallback. Add `?webgl` to the preview URL to force the fallback when testing.

## Asset credits

The following CC0 assets are from Poly Haven.

- [Kloofendal 48d Partly Cloudy (Pure Sky)](https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky)
- [Island Tree 01 leaf atlas](https://polyhaven.com/a/island_tree_01)
- [Fern 02](https://polyhaven.com/a/fern_02)
- [Dead Tree Trunk](https://polyhaven.com/a/dead_tree_trunk)
- [Rock Moss Set 01](https://polyhaven.com/a/rock_moss_set_01)
- [Forest Ground 01](https://polyhaven.com/a/forrest_ground_01)
- [Brown Mud Rocks 01](https://polyhaven.com/a/brown_mud_rocks_01)
- [Bark Brown 02](https://polyhaven.com/a/bark_brown_02)
- [Rock Boulder Dry](https://polyhaven.com/a/rock_boulder_dry)
- [Forest Slope](https://polyhaven.com/a/forest_slope)
