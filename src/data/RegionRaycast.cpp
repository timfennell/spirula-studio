// RegionRaycast.cpp -- first entry of camera rays into a region's voids: clip
// to the voids' bounding box, march at a fraction of its size, then bisect the
// first step that lands in a void.

#include "data/RegionRaycast.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

namespace spirula {

namespace {

Aabb intersect(const Aabb& a, const Aabb& b) {
    Aabb o;
    for (int k = 0; k < 3; k++) {
        o.lo[k] = std::max(a.lo[k], b.lo[k]);
        o.hi[k] = std::min(a.hi[k], b.hi[k]);
    }
    return o;
}

Aabb join(const Aabb& a, const Aabb& b) {
    if (a.unbounded() || b.unbounded()) return Aabb::everything();
    Aabb o = a;
    o.expand(b);
    return o;
}

// The lens maps undistorted (u, v) to distorted; invert it by fixed point.
void distort(const VoidRayCamera& c, double u, double v, double& du, double& dv) {
    const float* k = c.dist;
    const double r2 = u * u + v * v;
    if (c.distortion == 1) {
        const double radial = 1.0 + r2 * (k[0] + r2 * k[1]);
        du = u * radial + 2.0 * k[2] * u * v + k[3] * (r2 + 2.0 * u * u);
        dv = v * radial + 2.0 * k[3] * u * v + k[2] * (r2 + 2.0 * v * v);
    } else if (c.distortion == 2) {
        const double radial = 1.0 + r2 * (k[0] + r2 * (k[1] + r2 * (k[2] + r2 * k[3])));
        du = u * radial + 2.0 * k[4] * u * v + k[5] * (r2 + 2.0 * u * u) + k[6] * r2;
        dv = v * radial + 2.0 * k[5] * u * v + k[4] * (r2 + 2.0 * v * v) + k[7] * r2;
    } else {
        du = u;
        dv = v;
    }
}

void undistort(const VoidRayCamera& c, double xd, double yd, double& u, double& v) {
    u = xd;
    v = yd;
    if (c.distortion == 0) return;
    for (int it = 0; it < 20; it++) {
        double du, dv;
        distort(c, u, v, du, dv);
        u += xd - du;
        v += yd - dv;
    }
}

}  // namespace

Aabb region_void_bounds(const Region& r) {
    const auto* csg = dynamic_cast<const CsgRegion*>(&r);
    if (!csg) return Aabb::everything();
    switch (csg->op) {
        case CsgOp::Complement:
            return csg->children.empty() ? Aabb() : csg->children[0]->bounds();
        case CsgOp::Intersection: {   // outside any child
            Aabb b;
            for (const auto& c : csg->children) b = join(b, region_void_bounds(*c));
            return b;
        }
        case CsgOp::Union: {          // outside every child
            Aabb b = Aabb::everything();
            for (const auto& c : csg->children) b = intersect(b, region_void_bounds(*c));
            return b;
        }
        case CsgOp::Difference: {     // outside the first, or inside another
            if (csg->children.empty()) return Aabb();
            Aabb b = region_void_bounds(*csg->children[0]);
            for (size_t i = 1; i < csg->children.size(); i++) b = join(b, csg->children[i]->bounds());
            return b;
        }
    }
    return Aabb::everything();
}

void region_void_depth_map(const Region& r, double rs, const double center[3],
                           const VoidRayCamera& cam, int map_w, int map_h, float* out) {
    const float inf = std::numeric_limits<float>::infinity();
    std::fill(out, out + (size_t)map_w * map_h, inf);
    const Aabb box = region_void_bounds(r);
    if (box.empty() || box.unbounded() || map_w <= 0 || map_h <= 0) return;

    // Camera centre and axes, then everything in the dataset frame.
    const float* m = cam.viewmat;
    const double Rw[9] = {m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]};
    const double t[3] = {m[3], m[7], m[11]};
    double o[3];
    for (int k = 0; k < 3; k++)
        o[k] = -(Rw[k] * t[0] + Rw[3 + k] * t[1] + Rw[6 + k] * t[2]) / rs + center[k];
    double diag = 0;
    for (int k = 0; k < 3; k++) diag += (box.hi[k] - box.lo[k]) * (box.hi[k] - box.lo[k]);
    const double step = std::sqrt(diag) / 160.0;

    auto row = [&](int y) {
        for (int x = 0; x < map_w; x++) {
            const double px = (x + 0.5) * cam.width / map_w, py = (y + 0.5) * cam.height / map_h;
            double u, v;
            undistort(cam, (px - cam.intrin[2]) / cam.intrin[0], (py - cam.intrin[3]) / cam.intrin[1], u, v);
            // d: dataset-frame change per unit of camera z
            double d[3];
            for (int k = 0; k < 3; k++) d[k] = (Rw[k] * u + Rw[3 + k] * v + Rw[6 + k]) / rs;
            double s0 = 1e-9, s1 = 1e300;
            for (int k = 0; k < 3 && s0 <= s1; k++) {
                if (std::fabs(d[k]) < 1e-300) {
                    if (o[k] < box.lo[k] || o[k] > box.hi[k]) s1 = -1;
                    continue;
                }
                double a = (box.lo[k] - o[k]) / d[k], b = (box.hi[k] - o[k]) / d[k];
                if (a > b) std::swap(a, b);
                s0 = std::max(s0, a);
                s1 = std::min(s1, b);
            }
            if (s0 > s1) continue;
            const double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            const double ds = step / len;
            auto in_void = [&](double s) {
                const double p[3] = {o[0] + s * d[0], o[1] + s * d[1], o[2] + s * d[2]};
                return !r.contains(p);
            };
            double prev = s0;
            if (in_void(s0)) { out[(size_t)y * map_w + x] = (float)s0; continue; }
            for (double s = s0 + ds; s <= s1 + ds; s += ds) {
                if (!in_void(s)) { prev = s; continue; }
                double lo = prev, hi = s;
                for (int it = 0; it < 12; it++) {
                    const double mid = 0.5 * (lo + hi);
                    (in_void(mid) ? hi : lo) = mid;
                }
                out[(size_t)y * map_w + x] = (float)hi;
                break;
            }
        }
    };
    const int n = std::max(1, std::min<int>(map_h, (int)std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for (int w = 0; w < n; w++)
        pool.emplace_back([&, w] { for (int y = w; y < map_h; y += n) row(y); });
    for (auto& th : pool) th.join();
}

}  // namespace spirula
