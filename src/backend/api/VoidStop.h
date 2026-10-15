#pragma once

// "Voids block light": a void is opaque and shows noise. The next 2D raster
// forward stops each ray where it enters a void and lights the void's colour
// with what is left; its backward reads that light back. Armed by a training
// step only, so previews never see it. Vulkan only (shaders/void_noise.slang).

#include <cstdint>

namespace backend {

struct VoidStop {
    uint64_t depth = 0;   // device address of the void buffer (void_noise.slang), 0 = off
    uint32_t width = 0, height = 0;
};

inline VoidStop& void_stop() {
    static VoidStop s;
    return s;
}

// The same buffer for the training step's raster backward.
inline VoidStop& void_stop_bwd() {
    static VoidStop s;
    return s;
}

}  // namespace backend
