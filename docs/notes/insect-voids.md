# Voids inside shiny subjects

A glossy convex surface -- an insect's abdomen, a beetle's wing case -- mirrors
its lights as a virtual image about half its radius *behind* the surface.
Plain 3DGS reproduces that image literally: a cloud of bright splats inside the
body, seen through a half-transparent shell. Lowering the SH degree makes it
worse, since view-dependent colour is then the only other way to get the
highlight. This note covers the two tools that keep the body solid and the
shine on its surface: keep-out voids, and voids that block light.

## The workflow

Draw the voids on a quick draft, then train the real model with them from
step 0. Voids patched into a finished model clear the inside, but splats
already sunk under the surface where a void stops short stay put once
densification has ended.

1. **Draft.** Train only to establish the shapes: around 1M splats, half
   resolution, 10-15k steps, no region. It needs a clear outline of the shell
   in cross section, not detail.
2. **Draw.** In **Keep-out regions (insects)** open the **Cross-section void
   editor** (with nothing training it slices the newest saved run). It cuts the
   model into slices along the body's long, middle or short axis
   (`app/gui/VoidSlices.h`): each splat's cross section in the slice plane,
   base colour, faded by opacity. Slices are framed on the splats around where
   the cameras aim, up to the first clear gap, so a pin or mount does not
   stretch them. Draw each void as a blob: an outline on a few key slices,
   more where the body bends. Between keys the outline is interpolated (per
   blob, so separate voids never merge); it ends at its first and last key
   unless **Round the ends** is ticked. Bring every outline right up to the
   inner edge of the shell -- under fur that is the cuticle, not the base of
   the hair -- leaving a hair's width for the wall. **Save voids** writes
   `roi/voids.svoid` (the drawing) and `roi/voids.json` (the region: the
   complement of a stack of outline prisms, one per slice, in the dataset
   frame).
3. **Train for real** with the region and **Voids block light** from the
   first step: pick the voids file and **Use for the next run**.

The voids are in the dataset's frame, so every later run on the same dataset
reuses them; redraw only if the cameras are solved again. A paused run can
also take them (**Apply to paused run**), as long as densification still has
steps to go (below).

## Keep-out: splats inside a void fade

The region keeps splats out of the voids the way any region of interest does
(`--roi-region`): at every densification step a splat whose centre is in a void
keeps `--roi-outside-opacity-decay` (0.7) of its opacity and draws for growth
with `--roi-outside-weight` (1e-4). Ones the images still need win opacity
back; the rest fade and MCMC relocates them, and export drops anything under
1/255. Densification must run nearly to the end for this to keep acting, so the
panel raises `--refine-stop-iter` to 87.5% of the run. Pixel masking
(`--roi-mask-pixels`) is turned off: the voids are inside the subject, and
masking would hide the subject itself.

## Voids block light

Keep-out alone empties the interior, but it leaves the shell free to stay
half transparent: the near and far walls together still cover every pixel, so
the alpha losses are satisfied, and the shine moves onto the *inside* of the
far wall, seen through the near one. Cut the model in half and look at one
wall face-on, and the background shows through it.

`--roi-void-stop 1` ("Voids block light", default on in the panel) closes
that. Before each training forward, every camera ray stops where it first
enters a void:

- `data/RegionRaycast.h` casts the batch's cameras against the region on the
  CPU: clip to the voids' bounding box (`region_void_bounds`, from the CSG
  tree), march at 1/160 of its diagonal, bisect the first step that lands in a
  void. Lens distortion (OpenCV, thin prism) is undone per pixel. The map is
  512 px on its long side and is cached per camera pose, so after the first
  epoch it costs nothing.
- The engine uploads it (`fwd.void_depth`) and arms
  `backend::void_stop()` (`backend/api/VoidStop.h`) for that one forward; the
  Vulkan 2D raster (`rasterize_fwd_2d`, spec constant `kVoidStop`) ends a
  pixel's ray at the first splat whose depth is past the void.
- The backward needs no change: it replays each pixel from the forward's
  `last_ids`, which already stop short of the void.
- Light that gets through the near wall now meets the (random) background
  instead of the far wall, so the alpha loss makes the near wall opaque by
  itself, and the highlight has to live on its outer surface as SH.

Previews, the viewer and exports never see the stop: the training step arms it
immediately before its own forward and the forward disarms it.

Limits: the Vulkan backend and the 2D primitives (3dgs, mip) only -- the
trainer logs why and carries on without it otherwise; pinhole cameras (the
warp paths are skipped); the region must have closed voids (a plain region of
interest, whose outside reaches infinity, is refused).
`SS_VOID_STOP_DUMP=<dir>` writes each cast map as a PGM plus its camera, to
check the voids against the photos.

## Choosing settings

For a masked insect: random background, empty- and solid-area alpha weights
0.5, a small opacity regularization (0.001), SH degree 3, and full checkpoints
so the run can be paused and resumed around the void editing.
