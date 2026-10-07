// TrainCapture.cpp -- see TrainCapture.h.

#include "app/gui/TrainCapture.h"

#include "app/TrainerCore.h"
#include "app/gui/render/FrameSink.h"
#include "core/Camera.h"
#include "engine/Engine.h"
#include "external/stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace gui {

namespace {

TorchTensorView tvp(const void* p, uint32_t elem_size, std::vector<int64_t> shape) {
    return {(uint64_t)(uintptr_t)p, elem_size, std::move(shape)};
}
TorchTensorView tv_null() { return {0, 0, {}}; }

void cross(const float a[3], const float b[3], float o[3]) {
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}
float norm3(const float a[3]) { return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }
void normalize3(float a[3]) {
    const float n = norm3(a);
    if (n > 1e-12f) for (int i = 0; i < 3; i++) a[i] /= n;
}
void transpose3(const float a[9], float o[9]) {
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) o[c * 3 + r] = a[r * 3 + c];
}
void matmul3(const float a[9], const float b[9], float o[9]) {
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
}
void rot_axis(const float axis[3], float th, float o[9]) {
    const float c = std::cos(th), s = std::sin(th), t = 1 - c;
    const float x = axis[0], y = axis[1], z = axis[2];
    const float m[9] = {t * x * x + c,     t * x * y - s * z, t * x * z + s * y,
                        t * x * y + s * z, t * y * y + c,     t * y * z - s * x,
                        t * x * z - s * y, t * y * z + s * x, t * z * z + c};
    std::copy(m, m + 9, o);
}

void slerp(const double a[4], const double b_in[4], double t, double o[4]) {
    double b[4] = {b_in[0], b_in[1], b_in[2], b_in[3]};
    double d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if (d < 0) { for (double& v : b) v = -v; d = -d; }
    double wa = 1 - t, wb = t;
    if (d < 0.9995) {
        const double th = std::acos(d), s = std::sin(th);
        wa = std::sin((1 - t) * th) / s;
        wb = std::sin(t * th) / s;
    }
    double n = 0;
    for (int i = 0; i < 4; i++) { o[i] = wa * a[i] + wb * b[i]; n += o[i] * o[i]; }
    n = std::sqrt(n);
    for (int i = 0; i < 4; i++) o[i] /= n;
}

// Point nearest every camera's optical axis: where a turntable capture aims.
void axes_focus(const PostSplitCameras& post, float out[3]) {
    double A[9] = {0}, b[3] = {0};
    for (int64_t p = 0; p < post.n_post; p++) {
        const float* vm = &post.viewmats[(size_t)p * 16];
        const float R[9] = {vm[0], vm[1], vm[2], vm[4], vm[5], vm[6], vm[8], vm[9], vm[10]};
        const float t[3] = {vm[3], vm[7], vm[11]};
        double C[3], d[3] = {R[6], R[7], R[8]};
        for (int i = 0; i < 3; i++) C[i] = -(R[i] * t[0] + R[3 + i] * t[1] + R[6 + i] * t[2]);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) {
                const double m = (r == c ? 1.0 : 0.0) - d[r] * d[c];
                A[r * 3 + c] += m;
                b[r] += m * C[c];
            }
    }
    const double det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) +
                       A[2] * (A[3] * A[7] - A[4] * A[6]);
    if (std::abs(det) < 1e-12) return;
    const double inv[9] = {(A[4] * A[8] - A[5] * A[7]) / det, (A[2] * A[7] - A[1] * A[8]) / det,
                           (A[1] * A[5] - A[2] * A[4]) / det, (A[5] * A[6] - A[3] * A[8]) / det,
                           (A[0] * A[8] - A[2] * A[6]) / det, (A[2] * A[3] - A[0] * A[5]) / det,
                           (A[3] * A[7] - A[4] * A[6]) / det, (A[1] * A[6] - A[0] * A[7]) / det,
                           (A[0] * A[4] - A[1] * A[3]) / det};
    for (int r = 0; r < 3; r++)
        out[r] = (float)(inv[r * 3] * b[0] + inv[r * 3 + 1] * b[1] + inv[r * 3 + 2] * b[2]);
}

std::string safe_stem(const std::string& path) {
    std::string s = fs::path(path).stem().string();
    for (char& c : s)
        if (!(std::isalnum((unsigned char)c) || c == '-' || c == '_' || c == '.' || c == '+')) c = '_';
    return s;
}

}  // namespace


void capture_frame_size(int resolution, int& W, int& H) {
    switch (resolution) {
        case 2:  W = 3840; H = 2160; break;
        case 1:  W = 2560; H = 1440; break;
        default: W = 1920; H = 1080; break;
    }
}

std::vector<int> capture_schedule(const CaptureSettings& cs, int start, int total) {
    std::vector<int> steps;
    const int N = std::max(1, cs.every_steps);
    const int dense = std::max(1, N / 5);
    for (int st = start; st < total;) {
        steps.push_back(st);
        st += (cs.dense_start && st - start < 10 * N) ? dense : N;
    }
    if (total > start && steps.back() != total - 1) steps.push_back(total - 1);
    return steps;
}

namespace {

std::vector<std::string> split_names(const std::string& v) {
    std::vector<std::string> out;
    size_t a = 0;
    while (a <= v.size()) {
        const size_t b = v.find('|', a);
        const std::string s = v.substr(a, b == std::string::npos ? std::string::npos : b - a);
        if (!s.empty()) out.push_back(s);
        if (b == std::string::npos) break;
        a = b + 1;
    }
    return out;
}
std::string join_names(const std::vector<std::string>& v) {
    std::string s;
    for (const std::string& n : v) { if (!s.empty()) s += '|'; s += n; }
    return s;
}

}  // namespace

void read_capture_setting(CaptureSettings& cs, const std::string& k, const std::string& v) {
    if (k == "capture_enabled") cs.enabled = v != "0";
    else if (k == "capture_resolution") cs.resolution = std::clamp(std::atoi(v.c_str()), 0, 2);
    else if (k == "capture_every") cs.every_steps = std::max(1, std::atoi(v.c_str()));
    else if (k == "capture_dense") cs.dense_start = v != "0";
    else if (k == "capture_transparent") cs.transparent = v != "0";
    else if (k == "capture_fixed") cs.fixed_views = split_names(v);
    else if (k == "capture_arc") cs.arc = v != "0";
    else if (k == "capture_arc_views") cs.arc_views = split_names(v);
    else if (k == "capture_orbit") cs.orbit = v != "0";
    else if (k == "capture_orbit_view") cs.orbit_view = v;
    else if (k == "capture_orbit_degrees") cs.orbit_degrees = (float)std::atof(v.c_str());
    else if (k == "capture_output") cs.output = std::clamp(std::atoi(v.c_str()), 0, 2);
    else if (k == "capture_codec") cs.codec = std::clamp(std::atoi(v.c_str()), 0, 1);
    else if (k == "capture_quality") cs.quality = std::clamp(std::atoi(v.c_str()), 0, 2);
    else if (k == "capture_fps") cs.fps = std::max(1.0f, (float)std::atof(v.c_str()));
}

void write_capture_settings(FILE* f, const CaptureSettings& cs) {
    std::fprintf(f, "capture_enabled=%d\n", cs.enabled ? 1 : 0);
    std::fprintf(f, "capture_resolution=%d\n", cs.resolution);
    std::fprintf(f, "capture_every=%d\n", cs.every_steps);
    std::fprintf(f, "capture_dense=%d\n", cs.dense_start ? 1 : 0);
    std::fprintf(f, "capture_transparent=%d\n", cs.transparent ? 1 : 0);
    std::fprintf(f, "capture_fixed=%s\n", join_names(cs.fixed_views).c_str());
    std::fprintf(f, "capture_arc=%d\n", cs.arc ? 1 : 0);
    std::fprintf(f, "capture_arc_views=%s\n", join_names(cs.arc_views).c_str());
    std::fprintf(f, "capture_orbit=%d\n", cs.orbit ? 1 : 0);
    std::fprintf(f, "capture_orbit_view=%s\n", cs.orbit_view.c_str());
    std::fprintf(f, "capture_orbit_degrees=%.1f\n", cs.orbit_degrees);
    std::fprintf(f, "capture_output=%d\n", cs.output);
    std::fprintf(f, "capture_codec=%d\n", cs.codec);
    std::fprintf(f, "capture_quality=%d\n", cs.quality);
    std::fprintf(f, "capture_fps=%.2f\n", cs.fps);
}

TrainCapture::Pose TrainCapture::post_pose(const spirula::TrainerSession& s, int p) const {
    const auto& post = s.post;
    Pose o;
    const float* vm = &post.viewmats[(size_t)p * 16];
    const float R[9] = {vm[0], vm[1], vm[2], vm[4], vm[5], vm[6], vm[8], vm[9], vm[10]};
    std::copy(R, R + 9, o.R);
    for (int i = 0; i < 3; i++)
        o.C[i] = -(R[i] * vm[3] + R[3 + i] * vm[7] + R[6 + i] * vm[11]);
    for (int i = 0; i < 4; i++) o.intr[i] = post.intrins[(size_t)p * 4 + i];
    o.w = post.post_widths[(size_t)p];
    o.h = post.post_heights[(size_t)p];
    o.model = camera_model_to_string((CameraModelType)post.post_models[(size_t)p]);
    o.distortion = camera_distortion_to_string(
        (CameraDistortionType)post.post_distortions[(size_t)p]);
    for (int i = 0; i < 8; i++) o.dist[i] = post.dist_coeffs[(size_t)p * 8 + i];
    return o;
}

TrainCapture::Pose TrainCapture::track_pose(const spirula::TrainerSession& s,
                                            const Track& t, float u) const {
    if (t.kind == 0) return post_pose(s, t.posts[0]);
    if (t.kind == 2) {
        Pose o = post_pose(s, t.posts[0]);
        float Q[9], QT[9], R2[9];
        rot_axis(_up, u * _cs.orbit_degrees * (float)M_PI / 180.0f, Q);
        transpose3(Q, QT);
        matmul3(o.R, QT, R2);
        std::copy(R2, R2 + 9, o.R);
        float d[3] = {o.C[0] - _center[0], o.C[1] - _center[1], o.C[2] - _center[2]};
        for (int r = 0; r < 3; r++)
            o.C[r] = _center[r] + Q[r * 3] * d[0] + Q[r * 3 + 1] * d[1] + Q[r * 3 + 2] * d[2];
        o.distortion = "NONE";
        return o;
    }
    // Arc: keyframes in order, swung around the focus point at lerped distance.
    const int n = (int)t.posts.size();
    const float f = std::clamp(u, 0.0f, 1.0f) * (float)(n - 1);
    const int i = std::min((int)f, n - 2);
    const float a = f - (float)i;
    const Pose A = post_pose(s, t.posts[(size_t)i]), B = post_pose(s, t.posts[(size_t)i + 1]);
    Pose o = A;
    float da[3], db[3];
    for (int k = 0; k < 3; k++) { da[k] = A.C[k] - _center[k]; db[k] = B.C[k] - _center[k]; }
    const float la = norm3(da), lb = norm3(db);
    double va[4] = {0, da[0], da[1], da[2]}, vb[4] = {0, db[0], db[1], db[2]};
    // Direction slerp as a pure-vector quaternion: same great-circle arc.
    for (int k = 1; k < 4; k++) { va[k] /= std::max(la, 1e-12f); vb[k] /= std::max(lb, 1e-12f); }
    double v[4];
    slerp(va, vb, a, v);
    const float len = la + (lb - la) * a;
    for (int k = 0; k < 3; k++) o.C[k] = _center[k] + (float)v[k + 1] * len;
    // Aim at the lerp of each key's point nearest the focus, up lerped too:
    // slerping the keys' rotations separately lets the subject leave frame.
    auto aim = [&](const Pose& P, float T[3], float U[3]) {
        const float f[3] = {P.R[6], P.R[7], P.R[8]};
        const float d = (_center[0] - P.C[0]) * f[0] + (_center[1] - P.C[1]) * f[1] +
                        (_center[2] - P.C[2]) * f[2];
        for (int k = 0; k < 3; k++) { T[k] = P.C[k] + f[k] * d; U[k] = -P.R[3 + k]; }
    };
    float TA[3], UA[3], TB[3], UB[3], T[3], U[3];
    aim(A, TA, UA);
    aim(B, TB, UB);
    for (int k = 0; k < 3; k++) { T[k] = TA[k] + (TB[k] - TA[k]) * a; U[k] = UA[k] + (UB[k] - UA[k]) * a; }
    float z[3] = {T[0] - o.C[0], T[1] - o.C[1], T[2] - o.C[2]};
    normalize3(z);
    const float down[3] = {-U[0], -U[1], -U[2]};
    float x[3], y[3];
    cross(down, z, x);
    normalize3(x);
    cross(z, x, y);
    for (int k = 0; k < 3; k++) { o.R[k] = x[k]; o.R[3 + k] = y[k]; o.R[6 + k] = z[k]; }
    for (int k = 0; k < 4; k++) o.intr[k] = A.intr[k] + (B.intr[k] - A.intr[k]) * a;
    o.distortion = "NONE";
    return o;
}

void TrainCapture::begin(spirula::TrainerSession& s, const CaptureSettings& cs,
                         int start_step, int total_steps) {
    finish();
    _cs = cs;
    _tracks.clear();
    _steps.clear();
    _next = 0;
    if (!cs.enabled || !cs.any_track()) return;
    capture_frame_size(cs.resolution, _W, _H);

    const auto& names = s.ds.image_filenames;
    auto post_of = [&](const std::string& name) -> int {
        for (size_t i = 0; i < names.size(); i++)
            if (fs::path(names[i]).filename().string() == name)
                return s.post.post_offsets.empty() ? (int)i : s.post.post_offsets[i];
        return -1;
    };
    const fs::path root = s.out_dir / "captures";
    for (const std::string& v : cs.fixed_views) {
        const int p = post_of(v);
        if (p >= 0) _tracks.push_back({(root / ("view_" + safe_stem(v))).string(), {p}, 0});
    }
    if (cs.arc) {
        Track t{(root / "arc").string(), {}, 1};
        for (const std::string& v : cs.arc_views)
            if (int p = post_of(v); p >= 0) t.posts.push_back(p);
        if (t.posts.size() >= 2) _tracks.push_back(std::move(t));
    }
    if (cs.orbit) {
        if (int p = post_of(cs.orbit_view); p >= 0)
            _tracks.push_back({(root / "orbit").string(), {p}, 2});
    }
    if (_tracks.empty()) return;

    axes_focus(s.post, _center);
    float up[3] = {0, 0, 0};
    for (int64_t p = 0; p < s.post.n_post; p++)
        for (int k = 0; k < 3; k++) up[k] -= s.post.viewmats[(size_t)p * 16 + 4 + k];
    normalize3(up);
    std::copy(up, up + 3, _up);

    _steps = capture_schedule(cs, start_step, total_steps);
    if (_steps.empty()) return;

    _session = &s;
    for (Track& t : _tracks) {
        std::error_code ec;
        fs::create_directories(t.dir, ec);
        std::ofstream(fs::path(t.dir) / "frames.csv") << "frame,step\n";
        if (cs.output == 0) continue;
        render::Encoder e{render::Encoder::Ffmpeg, cs.ffmpeg, cs.codec == 0 ? "libx264" : "libx265"};
        const std::string mp4 =
            (fs::path(t.dir) / (fs::path(t.dir).filename().string() + ".mp4")).string();
        t.video = render::open_pipe_sink(
            render::encoder_argv(e, _W, _H, std::max(1.0f, cs.fps),
                                 cs.codec == 0 ? render::Codec::H264 : render::Codec::H265,
                                 cs.quality, false, mp4),
            _W, _H);
    }
    stbi_write_png_compression_level = 4;
    _closing = false;
    for (int i = 0; i < 2; i++) _writers.emplace_back([this] { writer_loop(); });
    char line[256];
    std::snprintf(line, sizeof line,
                  "Training capture: %d track(s), %d frames each at %dx%d (%s), into %s",
                  (int)_tracks.size(), (int)_steps.size(), _W, _H,
                  cs.output == 0 ? "PNG" : cs.output == 1 ? (cs.codec ? "H.265" : "H.264")
                                                          : (cs.codec ? "PNG + H.265" : "PNG + H.264"),
                  root.string().c_str());
    s.log(line);
}

void TrainCapture::render(spirula::TrainerSession& s, const Pose& p, Job& job) {
    const int W = _W, H = _H;
    const float sc = std::min((float)W / (float)p.w, (float)H / (float)p.h);
    const float intr[4] = {p.intr[0] * sc, p.intr[1] * sc,
                           0.5f * W + (p.intr[2] - 0.5f * p.w) * sc,
                           0.5f * H + (p.intr[3] - 0.5f * p.h) * sc};
    float vm[16] = {0};
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) vm[r * 4 + c] = p.R[r * 3 + c];
        vm[r * 4 + 3] = -(p.R[r * 3] * p.C[0] + p.R[r * 3 + 1] * p.C[1] + p.R[r * 3 + 2] * p.C[2]);
    }
    vm[15] = 1;
    const int64_t npx = (int64_t)W * H;
    _rgb.resize((size_t)npx * 3);
    _ts.resize((size_t)npx);
    {
        std::lock_guard<std::mutex> lk(s.engine_mutex);
        set_camera_params(W, H, p.model, p.distortion, tvp(vm, 4, {1, 4, 4}),
                          tvp(intr, 4, {1, 4}), tvp(p.dist, 4, {1, 8}));
        // The bare splats: training's random background would tint every
        // half-transparent edge with that step's noise.
        const bool bg = engine_set_background_enabled(false);
        try {
            forward_3dgs(s.cfg.primitive, 100, s.cfg.packed, false, 0);
            engine_copy_render_to_host(tvp(_rgb.data(), 4, {1, H, W, 3}), tv_null(),
                                       tvp(_ts.data(), 4, {1, H, W, 1}), tv_null(), tv_null());
        } catch (...) {
            engine_set_background_enabled(bg);
            throw;
        }
        engine_set_background_enabled(bg);
    }
    job.W = W;
    job.H = H;
    job.ch = _cs.transparent ? 4 : 3;
    const bool png = _cs.output != 1, video = _cs.output != 0;
    if (png) job.px.resize((size_t)npx * job.ch);
    if (video) job.rgb.resize((size_t)npx * 3);
    auto to8 = [](float v) { return (uint8_t)(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f); };
    for (int64_t i = 0; i < npx; i++) {
        const float a = std::clamp(1.0f - _ts[(size_t)i], 0.0f, 1.0f);
        for (int c = 0; c < 3; c++) {
            const float pre = std::min(_rgb[(size_t)i * 3 + c], a);
            if (video) job.rgb[(size_t)i * 3 + c] = to8(pre);
            // PNG alpha is straight, the render premultiplied.
            if (png) job.px[(size_t)i * job.ch + c] =
                to8(_cs.transparent ? (a > 1e-4f ? pre / a : 0.0f) : pre);
        }
        if (png && _cs.transparent) job.px[(size_t)i * 4 + 3] = to8(a);
    }
}

void TrainCapture::on_step(spirula::TrainerSession& s, int step) {
    if (_next >= _steps.size() || step < _steps[_next]) return;
    while (_next + 1 < _steps.size() && _steps[_next + 1] <= step) _next++;
    const int frame = (int)_next;
    const float u = _steps.size() > 1 ? (float)frame / (float)(_steps.size() - 1) : 0.0f;
    _next++;
    for (Track& t : _tracks) {
        Job job;
        char name[64];
        std::snprintf(name, sizeof name, "frame_%05d.png", frame);
        job.path = (fs::path(t.dir) / name).string();
        try {
            render(s, track_pose(s, t, u), job);
        } catch (const std::exception& e) {
            s.log(std::string("Training capture: render failed: ") + e.what());
            continue;
        }
        std::ofstream(fs::path(t.dir) / "frames.csv", std::ios::app) << frame << "," << step << "\n";
        if (t.video && !t.video->push(std::move(job.rgb))) {
            s.log("Training capture: video encoder stopped for " + t.dir + ": " + t.video->error());
            t.video->cancel();
            t.video.reset();
        }
        if (_cs.output == 1) continue;
        std::unique_lock<std::mutex> lk(_mu);
        _cv_room.wait(lk, [&] { return _queue.size() < 4; });
        _queue.push_back(std::move(job));
        _cv_job.notify_one();
    }
}

void TrainCapture::writer_loop() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(_mu);
            _cv_job.wait(lk, [&] { return _closing || !_queue.empty(); });
            if (_queue.empty()) return;
            job = std::move(_queue.front());
            _queue.pop_front();
        }
        _cv_room.notify_one();
        stbi_write_png(job.path.c_str(), job.W, job.H, job.ch, job.px.data(), job.W * job.ch);
    }
}

void TrainCapture::finish() {
    {
        std::lock_guard<std::mutex> lk(_mu);
        _closing = true;
    }
    _cv_job.notify_all();
    for (std::thread& t : _writers) if (t.joinable()) t.join();
    _writers.clear();
    _closing = false;
    for (Track& t : _tracks) {
        if (!t.video) continue;
        const bool ok = t.video->finish();
        if (_session)
            _session->log(ok ? "Training capture: wrote " + fs::path(t.dir).filename().string() + ".mp4"
                             : "Training capture: video failed for " + t.dir + ": " + t.video->error());
        t.video.reset();
    }
}

}  // namespace gui
