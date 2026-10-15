// The colour a void shows during training (backend/api/VoidStop.h): a cube
// corner per 64 px cell, the grid shifted every step, so light that leaks
// through a wall in front of a void lands on a colour no splat can predict.
// `strength` blends it from grey, ramped in like the random backgrounds.

[ForceInline]
uint _void_mix(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

[ForceInline]
float3 void_noise_color(float* buf, uint image_id, uint x, uint y) {
    const uint seed = asuint(buf[0]);
    const float strength = buf[1];
    const uint cell = 64u;
    uint cx = (x + _void_mix(seed * 2u + 1u) % cell) / cell;
    uint cy = (y + _void_mix(seed * 2u + 7u) % cell) / cell;
    uint h = _void_mix(cx * 2654435761u ^ cy * 40503u ^ (seed + image_id * 0x9e3779b9u));
    float3 corner = float3((h & 1u) != 0u ? 0.5f : -0.5f, (h & 2u) != 0u ? 0.5f : -0.5f,
                           (h & 4u) != 0u ? 0.5f : -0.5f);
    return 0.5f + strength * corner;
}

// Layout of the void buffer: [0] the step's seed (bits), [1] the strength,
// then the void depth maps [I, vh, vw], then the light left at each void
// pixel [I, H, W] (void_light_index).
[ForceInline]
uint void_map_index(uint image_id, uint vw, uint vh, uint x, uint y, uint W, uint H) {
    uint vx = min(x * vw / W, vw - 1u);
    uint vy = min(y * vh / H, vh - 1u);
    return 2u + (image_id * vh + vy) * vw + vx;
}

[ForceInline]
uint void_light_index(uint I, uint vw, uint vh, uint image_pix) {
    return 2u + I * vh * vw + image_pix;
}
