#pragma once

// "Voids block light": per-pixel camera depths past which the next 2D raster
// forward stops each ray. The engine fills it right before a training forward
// and clears it right after, so previews and viewer renders never see it.
// Only the Vulkan backend reads it (backend/vulkan/kernels/RasterizeFwd.cpp).

#include <cstdint>

namespace backend {

struct VoidStop {
    uint64_t depth = 0;   // device address of [I, height, width] floats, 0 = off
    uint32_t width = 0, height = 0;
};

inline VoidStop& void_stop() {
    static VoidStop s;
    return s;
}

}  // namespace backend
