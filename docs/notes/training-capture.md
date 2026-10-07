# Training animation capture

The trainer's **Training animation capture** panel records the model as it
trains, from cameras of its own, without touching the live preview
(`app/gui/TrainCapture.h`).

## Tracks

Any number at once, each to `<run>/captures/<track>/`:

- **Fixed views** -- one track per chosen training image, from that camera.
- **Arc** -- a path through two or more training views in order. Position, up
  vector and look-at target are interpolated between keyframes, so the camera
  stays on the subject instead of drifting with the cameras' own aim.
- **Orbit** -- a turn of `orbit_degrees` about the scene centre, starting at
  one training view, about the cameras' mean up.

A track's frames are spread over the run: frame k of n sits at u = k/(n-1)
along the arc or orbit, so the motion and the training finish together.

## Schedule and output

A frame every `every_steps` steps, five times as often over the first ten
intervals (`dense_start`), where the model changes most. HD, 2K or 4K. Output
is PNG frames (RGBA over nothing, or RGB over black), a video, or both. Video
is piped to ffmpeg (`render::open_pipe_sink`) as H.264 or H.265 at three
quality levels; a 4K H.265 sequence is about 250x smaller than its PNGs.
`frames.csv` maps each frame to its step.

## Cost

Frames are rendered on the training thread between steps, through the same
forward the trainer uses (`set_camera_params`, `forward_3dgs`,
`engine_copy_render_to_host`), so no second copy of the model exists on the
device; measured about 10 MB of extra GPU memory and 0.37 s per 4K frame on an
M2 Pro. PNG encoding and the ffmpeg pipe run on writer threads, off the
training thread. Settings persist in `gui.conf` as `capture_*` keys.
