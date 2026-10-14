// VoidSlices.cpp -- see VoidSlices.h.

#include "app/gui/VoidSlices.h"

#include "checkpoint/SplatPly.h"
#include "data/DatasetParser.h"
#include "data/Json.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace gui {

namespace {

constexpr double kC0 = 0.28209479177387814;
constexpr int kInterpPoints = 64;
constexpr int kRegionPoints = 40;

double dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
void cross3(const double* a, const double* b, double* o) {
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}
void norm3(double* a) {
    const double n = std::sqrt(dot3(a, a));
    if (n > 0) for (int i = 0; i < 3; i++) a[i] /= n;
}

// Symmetric 3x3 eigen-decomposition (Jacobi); columns of V, descending.
void eig3(double A[9], double w[3], double V[9]) {
    for (int i = 0; i < 9; i++) V[i] = i % 4 == 0 ? 1 : 0;
    for (int sweep = 0; sweep < 50; sweep++) {
        double off = std::fabs(A[1]) + std::fabs(A[2]) + std::fabs(A[5]);
        if (off < 1e-20) break;
        for (int p = 0; p < 2; p++)
            for (int q = p + 1; q < 3; q++) {
                const double apq = A[p * 3 + q];
                if (std::fabs(apq) < 1e-30) continue;
                const double th = 0.5 * std::atan2(2 * apq, A[q * 3 + q] - A[p * 3 + p]);
                const double c = std::cos(th), s = std::sin(th);
                for (int k = 0; k < 3; k++) {   // A = J^T A J
                    const double akp = A[k * 3 + p], akq = A[k * 3 + q];
                    A[k * 3 + p] = c * akp - s * akq;
                    A[k * 3 + q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; k++) {
                    const double apk = A[p * 3 + k], aqk = A[q * 3 + k];
                    A[p * 3 + k] = c * apk - s * aqk;
                    A[q * 3 + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; k++) {
                    const double vkp = V[k * 3 + p], vkq = V[k * 3 + q];
                    V[k * 3 + p] = c * vkp - s * vkq;
                    V[k * 3 + q] = s * vkp + c * vkq;
                }
            }
    }
    int idx[3] = {0, 1, 2};
    std::sort(idx, idx + 3, [&](int a, int b) { return A[a * 4] > A[b * 4]; });
    double V2[9];
    for (int j = 0; j < 3; j++) {
        w[j] = A[idx[j] * 4];
        for (int k = 0; k < 3; k++) V2[k * 3 + j] = V[k * 3 + idx[j]];
    }
    std::copy(V2, V2 + 9, V);
}

struct Splat {
    double mu[3];      // frame coords (u, v, n)
    double S[6];       // frame covariance: uu uv un vv vn nn
    float rgb[3], op;
};

std::vector<Splat> to_frame(const spirula::SplatCloud& c, double rs, const double center[3],
                            const VoidFrame* f) {
    std::vector<Splat> out((size_t)c.num);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < c.num; i++) {
        Splat& s = out[(size_t)i];
        double p[3];
        for (int k = 0; k < 3; k++) p[k] = c.means[(size_t)i * 3 + k] / rs + center[k];
        const float* q = &c.quats[(size_t)i * 4];
        double w = q[0], x = q[1], y = q[2], z = q[3];
        const double qn = std::sqrt(w * w + x * x + y * y + z * z);
        w /= qn; x /= qn; y /= qn; z /= qn;
        const double R[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                             2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                             2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)};
        double sc[3];
        for (int k = 0; k < 3; k++) sc[k] = std::exp(c.scales[(size_t)i * 3 + k]) / rs;
        double Sg[9];
        for (int r = 0; r < 3; r++)
            for (int cc = 0; cc < 3; cc++) {
                double v = 0;
                for (int k = 0; k < 3; k++) v += R[r * 3 + k] * sc[k] * sc[k] * R[cc * 3 + k];
                Sg[r * 3 + cc] = v;
            }
        if (f) {
            const double* B[3] = {f->u, f->v, f->n};
            const double d[3] = {p[0] - f->o[0], p[1] - f->o[1], p[2] - f->o[2]};
            for (int a = 0; a < 3; a++) s.mu[a] = dot3(B[a], d);
            double T[9];   // B^T Sg B
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) {
                    double v = 0;
                    for (int r = 0; r < 3; r++)
                        for (int cc = 0; cc < 3; cc++) v += B[a][r] * Sg[r * 3 + cc] * B[b][cc];
                    T[a * 3 + b] = v;
                }
            s.S[0] = T[0]; s.S[1] = T[1]; s.S[2] = T[2]; s.S[3] = T[4]; s.S[4] = T[5]; s.S[5] = T[8];
        } else {
            for (int k = 0; k < 3; k++) s.mu[k] = p[k];
        }
        for (int k = 0; k < 3; k++)
            s.rgb[k] = (float)std::clamp(c.features_dc[(size_t)i * 3 + k] * kC0 + 0.5, 0.0, 1.0);
        s.op = (float)(1.0 / (1.0 + std::exp(-(double)c.opacities[(size_t)i])));
    }
    return out;
}

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    const size_t k = std::min(v.size() - 1, (size_t)(q * (v.size() - 1)));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

std::vector<VPt> resample(const std::vector<VPt>& poly, int m) {
    const size_t n = poly.size();
    std::vector<double> cum(n + 1, 0);
    for (size_t i = 0; i < n; i++) {
        const VPt& a = poly[i]; const VPt& b = poly[(i + 1) % n];
        cum[i + 1] = cum[i] + std::hypot(b[0] - a[0], b[1] - a[1]);
    }
    std::vector<VPt> out((size_t)m);
    const double L = cum[n];
    size_t seg = 0;
    for (int k = 0; k < m; k++) {
        const double s = L * k / m;
        while (seg + 1 < n && cum[seg + 1] < s) seg++;
        const VPt& a = poly[seg]; const VPt& b = poly[(seg + 1) % n];
        const double len = cum[seg + 1] - cum[seg];
        const double t = len > 0 ? (s - cum[seg]) / len : 0;
        out[(size_t)k] = {a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t};
    }
    return out;
}

double signed_area(const std::vector<VPt>& p) {
    double a = 0;
    for (size_t i = 0; i < p.size(); i++) {
        const VPt& u = p[i]; const VPt& w = p[(i + 1) % p.size()];
        a += u[0] * w[1] - w[0] * u[1];
    }
    return 0.5 * a;
}

// Closed Catmull-Rom through the clicked corners: a few clicks, a rounded outline.
std::vector<VPt> smooth_closed(const std::vector<VPt>& p, int per_seg = 8) {
    const size_t n = p.size();
    if (n < 3) return p;
    std::vector<VPt> out;
    out.reserve(n * per_seg);
    for (size_t i = 0; i < n; i++) {
        const VPt& p0 = p[(i + n - 1) % n]; const VPt& p1 = p[i];
        const VPt& p2 = p[(i + 1) % n];     const VPt& p3 = p[(i + 2) % n];
        for (int s = 0; s < per_seg; s++) {
            const double t = (double)s / per_seg, t2 = t * t, t3 = t2 * t;
            VPt q;
            for (int c = 0; c < 2; c++)
                q[c] = 0.5 * (2 * p1[c] + (-p0[c] + p2[c]) * t + (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * t2 +
                              (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * t3);
            out.push_back(q);
        }
    }
    return out;
}

// How many slices a rounded end cap spans: the outline's mean radius.
int cap_slices(const std::vector<VPt>& p, double pitch) {
    const double r = std::sqrt(std::fabs(signed_area(p)) / M_PI);
    return std::max(1, (int)std::lround(r / pitch));
}

std::vector<VPt> shrink(const std::vector<VPt>& p, double s) {
    double cx = 0, cy = 0;
    for (const VPt& q : p) { cx += q[0]; cy += q[1]; }
    cx /= p.size(); cy /= p.size();
    std::vector<VPt> o(p.size());
    for (size_t i = 0; i < p.size(); i++) o[i] = {cx + (p[i][0] - cx) * s, cy + (p[i][1] - cy) * s};
    return o;
}

std::vector<VPt> interpolate(std::vector<VPt> a, std::vector<VPt> b, double t) {
    if (signed_area(a) < 0) std::reverse(a.begin(), a.end());
    if (signed_area(b) < 0) std::reverse(b.begin(), b.end());
    a = resample(a, kInterpPoints);
    b = resample(b, kInterpPoints);
    int best = 0;
    double bd = 1e300;
    for (int s = 0; s < kInterpPoints; s++) {
        double d = 0;
        for (int k = 0; k < kInterpPoints; k++) {
            const VPt& p = a[(size_t)k]; const VPt& q = b[(size_t)((k + s) % kInterpPoints)];
            d += (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]);
        }
        if (d < bd) { bd = d; best = s; }
    }
    std::vector<VPt> out((size_t)kInterpPoints);
    for (int k = 0; k < kInterpPoints; k++) {
        const VPt& p = a[(size_t)k]; const VPt& q = b[(size_t)((k + best) % kInterpPoints)];
        out[(size_t)k] = {p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t};
    }
    return out;
}

void put(std::string& s, double x) {
    char b[40];
    std::snprintf(b, sizeof b, "%.17g", x);
    s += b;
}

}  // namespace


VoidFrame make_void_frame(const spirula::SplatCloud& c, double rs, const double center[3],
                          const double focus[3], int axis, int slices, int long_px) {
    VoidFrame f;
    f.axis = axis;
    std::vector<Splat> sp = to_frame(c, rs, center, nullptr);
    {
        // Distances from the focus; the subject ends at the first run of empty bins.
        std::vector<double> d;
        for (const Splat& s : sp)
            if (s.op > 0.2f)
                d.push_back(std::sqrt((s.mu[0] - focus[0]) * (s.mu[0] - focus[0]) +
                                      (s.mu[1] - focus[1]) * (s.mu[1] - focus[1]) +
                                      (s.mu[2] - focus[2]) * (s.mu[2] - focus[2])));
        if (d.empty()) throw std::runtime_error("the model has no opaque splats to slice");
        const double med = percentile(d, 0.5), bw = std::max(med / 8, 1e-12);
        const int nb = 64;
        std::vector<int> hist(nb, 0);
        for (double x : d) if (x < bw * nb) hist[(size_t)(x / bw)]++;
        // "Empty" allows the odd stray splat between subject and mount.
        const int peak = *std::max_element(hist.begin(), hist.end());
        const int few = std::max(2, peak / 300);
        double R = bw * nb;
        for (int k = (int)(med / bw); k + 2 < nb; k++)
            if (hist[k] < few && hist[k + 1] < few && hist[k + 2] < few) { R = k * bw; break; }
        for (Splat& s : sp) {
            const double r2 = (s.mu[0] - focus[0]) * (s.mu[0] - focus[0]) + (s.mu[1] - focus[1]) * (s.mu[1] - focus[1]) +
                              (s.mu[2] - focus[2]) * (s.mu[2] - focus[2]);
            if (r2 > R * R) s.op = 0;
        }
    }
    double m[3] = {0, 0, 0}, wsum = 0;
    for (const Splat& s : sp)
        if (s.op > 0.2f) { for (int k = 0; k < 3; k++) m[k] += s.op * s.mu[k]; wsum += s.op; }
    if (wsum <= 0) throw std::runtime_error("the model has no opaque splats to slice");
    for (double& x : m) x /= wsum;
    double C[9] = {0}, w[3], V[9];
    for (const Splat& s : sp) {
        if (s.op <= 0.2f) continue;
        const double d[3] = {s.mu[0] - m[0], s.mu[1] - m[1], s.mu[2] - m[2]};
        for (int r = 0; r < 3; r++) for (int cc = 0; cc < 3; cc++) C[r * 3 + cc] += s.op * d[r] * d[cc];
    }
    eig3(C, w, V);
    double ax[3][3];
    for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++) ax[j][k] = axis < 3 ? V[k * 3 + j] : (k == j ? 1.0 : 0.0);
    const int ni = axis % 3;
    std::copy(ax[ni], ax[ni] + 3, f.n);
    std::copy(ax[(ni + 1) % 3], ax[(ni + 1) % 3] + 3, f.u);
    cross3(f.n, f.u, f.v);
    norm3(f.v);
    std::copy(m, m + 3, f.o);

    std::vector<double> pu, pv, pn;
    for (const Splat& s : sp) {
        if (s.op <= 0.2f) continue;
        const double d[3] = {s.mu[0] - m[0], s.mu[1] - m[1], s.mu[2] - m[2]};
        pu.push_back(dot3(f.u, d)); pv.push_back(dot3(f.v, d)); pn.push_back(dot3(f.n, d));
    }
    auto span = [](const std::vector<double>& v, double& lo, double& hi) {
        lo = percentile(v, 0.005); hi = percentile(v, 0.995);
        const double pad = 0.06 * (hi - lo);
        lo -= pad; hi += pad;
    };
    double ulo, uhi, vlo, vhi, nlo, nhi;
    span(pu, ulo, uhi); span(pv, vlo, vhi); span(pn, nlo, nhi);
    f.slices = std::max(4, slices);
    f.t0 = nlo;
    f.pitch = (nhi - nlo) / f.slices;
    f.px = std::max(uhi - ulo, vhi - vlo) / std::max(64, long_px);
    f.u0 = ulo; f.v0 = vlo;
    f.W = std::max(1, (int)std::ceil((uhi - ulo) / f.px));
    f.H = std::max(1, (int)std::ceil((vhi - vlo) / f.px));
    return f;
}

VoidFrame refine_void_frame(const VoidFrame& f, int slices, int long_px) {
    VoidFrame g = f;
    const double eu = f.W * f.px, ev = f.H * f.px;
    g.slices = std::max(4, slices);
    g.pitch = f.slices * f.pitch / g.slices;
    g.px = std::max(eu, ev) / std::max(64, long_px);
    g.W = std::max(1, (int)std::ceil(eu / g.px));
    g.H = std::max(1, (int)std::ceil(ev / g.px));
    return g;
}

void restack_blobs(VoidDoc& d, const VoidFrame& to) {
    for (VoidBlob& b : d.blobs) {
        VoidBlob nb = b;
        nb.keys.clear();
        for (const VoidKey& k : b.keys) {
            const double t = d.f.t0 + (k.slice + 0.5) * d.f.pitch;
            const int s = (int)std::lround((t - to.t0) / to.pitch - 0.5);
            set_key(nb, std::clamp(s, 0, to.slices - 1), k.pts);
        }
        b = std::move(nb);
    }
    d.f = to;
}

std::vector<std::vector<unsigned char>> render_void_slices(const spirula::SplatCloud& c, double rs,
                                                           const double center[3], const VoidFrame& f) {
    const std::vector<Splat> sp = to_frame(c, rs, center, &f);
    std::vector<std::vector<int>> bins((size_t)f.slices);
    for (size_t i = 0; i < sp.size(); i++) {
        const Splat& s = sp[i];
        if (s.op < 0.01f) continue;
        const double sn = std::sqrt(std::max(s.S[5], 1e-30)), r = 3 * sn + 0.5 * f.pitch;
        const int a = std::max(0, (int)std::floor((s.mu[2] - r - f.t0) / f.pitch));
        const int b = std::min(f.slices - 1, (int)std::floor((s.mu[2] + r - f.t0) / f.pitch));
        for (int k = a; k <= b; k++) bins[(size_t)k].push_back((int)i);
    }
    std::vector<std::vector<unsigned char>> out((size_t)f.slices);
    const double floor_var = 0.25 * f.px * f.px;
#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < f.slices; k++) {
        std::vector<float> acc((size_t)f.W * f.H * 3, 0.0f), wsum((size_t)f.W * f.H, 0.0f);
        const double t = f.t0 + (k + 0.5) * f.pitch;
        for (int i : bins[(size_t)k]) {
            const Splat& s = sp[(size_t)i];
            const double snn = std::max(s.S[5], 1e-30), dn = t - s.mu[2];
            const double a0 = s.op * std::exp(-0.5 * dn * dn / snn);
            if (a0 < 1e-3) continue;
            const double mu_u = s.mu[0] + s.S[2] * dn / snn, mu_v = s.mu[1] + s.S[4] * dn / snn;
            double cuu = s.S[0] - s.S[2] * s.S[2] / snn + floor_var;
            double cvv = s.S[3] - s.S[4] * s.S[4] / snn + floor_var;
            double cuv = s.S[1] - s.S[2] * s.S[4] / snn;
            const double det = cuu * cvv - cuv * cuv;
            if (det <= 0) continue;
            const double iuu = cvv / det, ivv = cuu / det, iuv = -cuv / det;
            const double ru = 3 * std::sqrt(cuu), rv = 3 * std::sqrt(cvv);
            const int x0 = std::max(0, (int)std::floor((mu_u - ru - f.u0) / f.px));
            const int x1 = std::min(f.W - 1, (int)std::floor((mu_u + ru - f.u0) / f.px));
            const int y0 = std::max(0, (int)std::floor((mu_v - rv - f.v0) / f.px));
            const int y1 = std::min(f.H - 1, (int)std::floor((mu_v + rv - f.v0) / f.px));
            for (int y = y0; y <= y1; y++) {
                const double dv = f.v0 + (y + 0.5) * f.px - mu_v;
                for (int x = x0; x <= x1; x++) {
                    const double du = f.u0 + (x + 0.5) * f.px - mu_u;
                    const double md = iuu * du * du + 2 * iuv * du * dv + ivv * dv * dv;
                    if (md > 9) continue;
                    const float a = (float)(a0 * std::exp(-0.5 * md));
                    const size_t p = (size_t)y * f.W + x;
                    wsum[p] += a;
                    for (int c3 = 0; c3 < 3; c3++) acc[p * 3 + c3] += a * s.rgb[c3];
                }
            }
        }
        std::vector<unsigned char> img((size_t)f.W * f.H * 3);
        for (size_t p = 0; p < wsum.size(); p++) {
            const float cov = 1.0f - std::exp(-wsum[p]);
            for (int c3 = 0; c3 < 3; c3++) {
                const float col = wsum[p] > 0 ? acc[p * 3 + c3] / wsum[p] : 0.0f;
                const float v = 0.07f * (1 - cov) + col * cov;
                img[p * 3 + c3] = (unsigned char)std::clamp(v * 255.0f + 0.5f, 0.0f, 255.0f);
            }
        }
        out[(size_t)k] = std::move(img);
    }
    return out;
}

std::vector<VPt> blob_outline(const VoidBlob& b, int slice, double pitch, bool* is_key) {
    if (is_key) *is_key = false;
    if (b.keys.empty()) return {};
    const VoidKey* lo = nullptr;
    const VoidKey* hi = nullptr;
    for (const VoidKey& k : b.keys) {
        if (k.slice == slice) { if (is_key) *is_key = true; return smooth_closed(k.pts); }
        if (k.slice < slice) lo = &k;
        if (k.slice > slice && !hi) hi = &k;
    }
    if (lo && hi)
        return interpolate(smooth_closed(lo->pts), smooth_closed(hi->pts),
                           (double)(slice - lo->slice) / (hi->slice - lo->slice));
    // Past an end key: an elliptical cap closing on the outline's centre.
    if (!b.round_ends) return {};
    const VoidKey& end = lo ? *lo : *hi;
    if (end.pts.size() < 3 || pitch <= 0) return {};
    const std::vector<VPt> e = smooth_closed(end.pts);
    const int L = cap_slices(e, pitch), d = std::abs(slice - end.slice);
    if (d > L) return {};
    const double x = (double)d / (L + 1);
    return shrink(e, std::sqrt(1 - x * x));
}

void blob_span(const VoidBlob& b, double pitch, int& first, int& last) {
    first = last = 0;
    if (b.keys.empty()) return;
    first = b.keys.front().slice;
    last = b.keys.back().slice;
    if (!b.round_ends) return;
    first -= cap_slices(smooth_closed(b.keys.front().pts), pitch);
    last += cap_slices(smooth_closed(b.keys.back().pts), pitch);
}

void set_key(VoidBlob& b, int slice, std::vector<VPt> pts) {
    b.keys.erase(std::remove_if(b.keys.begin(), b.keys.end(),
                                [&](const VoidKey& k) { return k.slice == slice; }),
                 b.keys.end());
    if (pts.size() >= 3) b.keys.push_back({slice, std::move(pts)});
    std::sort(b.keys.begin(), b.keys.end(), [](const VoidKey& a, const VoidKey& c) { return a.slice < c.slice; });
}

void cameras_focus(const ParsedDataset& ds, double rs, const double center[3], double out[3]) {
    double A[9] = {0}, b[3] = {0};
    for (int64_t i = 0; i < ds.num_cameras; i++) {
        const float* m = &ds.c2w[(size_t)i * 12];
        double C[3], d[3];
        for (int k = 0; k < 3; k++) { C[k] = m[k * 4 + 3] / rs + center[k]; d[k] = -m[k * 4 + 2]; }
        for (int r = 0; r < 3; r++)
            for (int cc = 0; cc < 3; cc++) {
                const double mm = (r == cc ? 1.0 : 0.0) - d[r] * d[cc];
                A[r * 3 + cc] += mm; b[r] += mm * C[cc];
            }
    }
    const double det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) +
                       A[2] * (A[3] * A[7] - A[4] * A[6]);
    if (std::fabs(det) < 1e-18) { std::copy(center, center + 3, out); return; }
    const double inv[9] = {(A[4] * A[8] - A[5] * A[7]) / det, (A[2] * A[7] - A[1] * A[8]) / det,
                           (A[1] * A[5] - A[2] * A[4]) / det, (A[5] * A[6] - A[3] * A[8]) / det,
                           (A[0] * A[8] - A[2] * A[6]) / det, (A[2] * A[3] - A[0] * A[5]) / det,
                           (A[3] * A[7] - A[4] * A[6]) / det, (A[1] * A[6] - A[0] * A[7]) / det,
                           (A[0] * A[4] - A[1] * A[3]) / det};
    for (int r = 0; r < 3; r++) out[r] = inv[r * 3] * b[0] + inv[r * 3 + 1] * b[1] + inv[r * 3 + 2] * b[2];
}

std::string void_region_json(const VoidDoc& d) {
    const VoidFrame& f = d.f;
    std::string prisms;
    int count = 0;
    for (const VoidBlob& b : d.blobs) {
        if (b.keys.empty()) continue;
        int first, last;
        blob_span(b, f.pitch, first, last);
        for (int s = std::max(0, first); s <= std::min(f.slices - 1, last); s++) {
            std::vector<VPt> poly = blob_outline(b, s, f.pitch);
            if (poly.size() < 3) continue;
            if (signed_area(poly) < 0) std::reverse(poly.begin(), poly.end());
            if ((int)poly.size() > kRegionPoints) poly = resample(poly, kRegionPoints);
            const double t = f.t0 + (s + 0.5) * f.pitch;
            if (count++) prisms += ",";
            prisms += "{\"type\":\"prism\",\"center\":[";
            for (int k = 0; k < 3; k++) { if (k) prisms += ","; put(prisms, f.o[k] + f.n[k] * t); }
            prisms += "],\"rotation\":[";
            const double* rows[3] = {f.u, f.v, f.n};
            for (int r = 0; r < 3; r++)
                for (int k = 0; k < 3; k++) { if (r || k) prisms += ","; put(prisms, rows[r][k]); }
            // Slabs overlap a hair so the stack has no gaps between them.
            prisms += "],\"half_height\":";
            put(prisms, 0.51 * f.pitch);
            prisms += ",\"polygon\":[";
            for (size_t i = 0; i < poly.size(); i++) {
                if (i) prisms += ",";
                put(prisms, poly[i][0]); prisms += ","; put(prisms, poly[i][1]);
            }
            prisms += "]}";
        }
    }
    if (!count) return {};
    return "{\"type\":\"complement\",\"children\":[{\"type\":\"union\",\"children\":[" + prisms + "]}]}";
}

void save_void_doc(const VoidDoc& d, const std::string& path) {
    const VoidFrame& f = d.f;
    std::string s = "{\"spirula_voids\":1,\"source\":\"";
    for (char c : d.source) { if (c == '"' || c == '\\') s += '\\'; s += c; }
    s += "\",\"frame\":{";
    auto vec = [&](const char* k, const double* v, int n) {
        s += "\""; s += k; s += "\":[";
        for (int i = 0; i < n; i++) { if (i) s += ","; put(s, v[i]); }
        s += "],";
    };
    vec("o", f.o, 3); vec("u", f.u, 3); vec("v", f.v, 3); vec("n", f.n, 3);
    const double nums[6] = {f.t0, f.pitch, f.u0, f.v0, f.px, (double)f.axis};
    vec("t0_pitch_u0_v0_px_axis", nums, 6);
    s += "\"slices\":" + std::to_string(f.slices) + ",\"W\":" + std::to_string(f.W) +
         ",\"H\":" + std::to_string(f.H) + "},\"blobs\":[";
    for (size_t bi = 0; bi < d.blobs.size(); bi++) {
        const VoidBlob& b = d.blobs[bi];
        if (bi) s += ",";
        s += "{\"name\":\"" + b.name + "\",\"round\":" + (b.round_ends ? "true" : "false") + ",\"keys\":[";
        for (size_t ki = 0; ki < b.keys.size(); ki++) {
            if (ki) s += ",";
            s += "{\"slice\":" + std::to_string(b.keys[ki].slice) + ",\"pts\":[";
            for (size_t i = 0; i < b.keys[ki].pts.size(); i++) {
                if (i) s += ",";
                put(s, b.keys[ki].pts[i][0]); s += ","; put(s, b.keys[ki].pts[i][1]);
            }
            s += "]}";
        }
        s += "]}";
    }
    s += "]}\n";
    std::ofstream(path) << s;
}

bool load_void_doc(const std::string& path, VoidDoc& d, std::string& error) {
    try {
        const JsonValue v = json_parse_file(path);
        const JsonValue* fr = v.find("frame");
        if (!fr) { error = "not a void drawing"; return false; }
        auto rd = [&](const char* k, double* out, size_t n) {
            const JsonValue* a = fr->find(k);
            if (!a || !a->is_array() || a->arr.size() < n) throw std::runtime_error(std::string("missing ") + k);
            for (size_t i = 0; i < n; i++) out[i] = a->arr[i].as_double();
        };
        VoidDoc o;
        rd("o", o.f.o, 3); rd("u", o.f.u, 3); rd("v", o.f.v, 3); rd("n", o.f.n, 3);
        double nums[6];
        rd("t0_pitch_u0_v0_px_axis", nums, 6);
        o.f.t0 = nums[0]; o.f.pitch = nums[1]; o.f.u0 = nums[2]; o.f.v0 = nums[3]; o.f.px = nums[4];
        o.f.axis = (int)nums[5];
        o.f.slices = (int)fr->get_double("slices", 0);
        o.f.W = (int)fr->get_double("W", 0);
        o.f.H = (int)fr->get_double("H", 0);
        if (const JsonValue* s = v.find("source")) o.source = s->as_string();
        if (const JsonValue* bs = v.find("blobs"); bs && bs->is_array())
            for (const JsonValue& bv : bs->arr) {
                VoidBlob b;
                if (const JsonValue* n = bv.find("name")) b.name = n->as_string();
                if (const JsonValue* r = bv.find("round")) b.round_ends = r->as_bool();
                if (const JsonValue* ks = bv.find("keys"); ks && ks->is_array())
                    for (const JsonValue& kv : ks->arr) {
                        VoidKey k;
                        k.slice = (int)kv.get_double("slice", 0);
                        if (const JsonValue* p = kv.find("pts"); p && p->is_array())
                            for (size_t i = 0; i + 1 < p->arr.size(); i += 2)
                                k.pts.push_back({p->arr[i].as_double(), p->arr[i + 1].as_double()});
                        b.keys.push_back(std::move(k));
                    }
                o.blobs.push_back(std::move(b));
            }
        d = std::move(o);
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

}  // namespace gui
