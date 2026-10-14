#pragma once

// Where camera rays first enter a region's voids -- the points it leaves out.
// Training can stop each ray there ("voids block light"), so a shell around a
// void has to be opaque on its own instead of leaning on the wall behind it.

#include "data/Region.h"

#include <cstdint>
#include <string>
#include <vector>

namespace spirula {

// Where `contains` can be false. Unbounded (Aabb::everything) when the voids
// reach infinity, as for a plain region of interest.
Aabb region_void_bounds(const Region& r);

// One camera as the trainer holds it: world-to-camera rows (training frame),
// fx fy cx cy in pixels, and the lens tier's coefficients (core/CameraModel.h).
struct VoidRayCamera {
    float viewmat[16];
    float intrin[4];
    float dist[8];
    int distortion = 0;   // CameraDistortionType: 0 none, 1 OpenCV, 2 thin prism
    int width = 0, height = 0;
};

// Camera depth (z) at which each ray of a map_w x map_h grid spanning the
// image first enters a void, or +inf where it misses. The region is in the
// dataset frame: p_dataset = p_train / rs + center. Pinhole cameras only.
void region_void_depth_map(const Region& r, double rs, const double center[3],
                           const VoidRayCamera& cam, int map_w, int map_h, float* out);

}  // namespace spirula
