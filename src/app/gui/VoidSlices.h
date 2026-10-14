#pragma once

// VoidSlices -- cross sections of a splat model, and void blobs traced on
// them. A blob is outlined on a few key slices and interpolated between, per
// blob, so separate voids never merge. The voids become a keep-out region: the
// complement of a stack of slab-thick outline prisms, in the dataset frame.

#include <array>
#include <string>
#include <vector>

namespace spirula { struct SplatCloud; }
struct ParsedDataset;

namespace gui {

using VPt = std::array<double, 2>;   // (u, v) in dataset units

// Slice k is the slab around t = t0 + (k + 0.5) * pitch along n.
struct VoidFrame {
    double o[3] = {0, 0, 0}, u[3] = {1, 0, 0}, v[3] = {0, 1, 0}, n[3] = {0, 0, 1};
    double t0 = 0, pitch = 1;
    int slices = 0;
    double u0 = 0, v0 = 0, px = 1;   // image pixel (x, y) is at u0 + (x+.5)px, v0 + (y+.5)px
    int W = 0, H = 0;
    int axis = 0;                    // 0..2 body long/mid/short, 3..5 dataset X/Y/Z
};

struct VoidKey { int slice = 0; std::vector<VPt> pts; };
// keys sorted by slice; the blob spans its first to last key, and only rounds
// off past them when `round_ends` is set.
struct VoidBlob { std::string name; std::vector<VoidKey> keys; bool round_ends = false; };

struct VoidDoc {
    VoidFrame f;
    std::string source;              // the splat.ply the slices came from
    std::vector<VoidBlob> blobs;
};

// p_dataset = p_train / rs + center. The frame keeps to the splats around
// `focus` (where the cameras aim) up to the first clear gap, so a separate
// mount or background does not stretch the slices.
VoidFrame make_void_frame(const spirula::SplatCloud& c, double rs, const double center[3],
                          const double focus[3], int axis, int slices, int long_px);
// One RGB image per slice: each splat's cross section in the slice plane,
// base colour, faded by opacity and by how far its centre is off the plane.
std::vector<std::vector<unsigned char>> render_void_slices(const spirula::SplatCloud& c, double rs,
                                                           const double center[3], const VoidFrame& f);

// The blob's smoothed outline on `slice`: its key there, else interpolated
// between the keys either side, else (round_ends) a rounded cap past an end.
std::vector<VPt> blob_outline(const VoidBlob& b, int slice, double pitch, bool* is_key = nullptr);
// First and last slice the blob reaches, caps included.
void blob_span(const VoidBlob& b, double pitch, int& first, int& last);
void set_key(VoidBlob& b, int slice, std::vector<VPt> pts);   // empty pts removes it

// The same frame recut into `slices` slices, `long_px` on the long side, so
// blobs keep their place; restack_blobs moves each key to the slice now at
// its depth and adopts the new frame.
VoidFrame refine_void_frame(const VoidFrame& f, int slices, int long_px);
void restack_blobs(VoidDoc& d, const VoidFrame& to);

// Where the dataset's cameras aim (least squares over their optical axes), in
// the dataset frame. ds holds the training frame (see make_void_frame).
void cameras_focus(const ParsedDataset& ds, double rs, const double center[3], double out[3]);

std::string void_region_json(const VoidDoc& d);
void save_void_doc(const VoidDoc& d, const std::string& path);
bool load_void_doc(const std::string& path, VoidDoc& d, std::string& error);

}  // namespace gui
