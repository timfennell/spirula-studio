#pragma once

// VoidEditor -- a floating window over the trainer: the model's cross
// sections, and void blobs traced on them (VoidSlices). Saves the drawing as
// <dataset>/roi/voids.svoid and the keep-out region as <dataset>/roi/voids.json.

#include "app/gui/GlLoader.h"
#include "app/gui/VoidSlices.h"

#include <future>
#include <string>
#include <vector>

namespace gui {

class VoidEditor {
public:
    ~VoidEditor();
    // `ply` is the model to slice (a splat.ply or a run/checkpoint folder);
    // rs / center take its training frame to the dataset frame.
    void open(const std::string& dataset_dir, const std::string& ply, double rs, const double center[3],
              const double focus[3]);
    bool is_open() const { return _open; }
    void draw();
    void destroy_gl();
    // The region file the user asked to apply to the paused run; "" if none.
    std::string take_apply_request();
    // Whether the trainer can take one now (set by the app every frame).
    void set_can_apply(bool can, const std::string& why) { _can_apply = can; _apply_why = why; }
    void set_note(const std::string& s) { _note = s; }

private:
    struct Gen { VoidDoc doc; std::vector<std::vector<unsigned char>> imgs; std::string error; };
    void start(bool new_frame);
    void save();
    void draw_controls();
    void draw_canvas();

    bool _open = false;
    std::string _dataset, _ply;
    double _rs = 1, _center[3] = {0, 0, 0}, _focus[3] = {0, 0, 0};
    int _axis = 0, _slices = 96, _res = 640;

    std::future<Gen> _gen;
    VoidDoc _doc;
    std::vector<std::vector<unsigned char>> _imgs;
    std::string _error, _note, _apply_req, _apply_why;
    bool _can_apply = false, _dirty = false;

    int _slice = 0, _blob = -1;
    std::vector<VPt> _drawing;      // a polygon being clicked out, (u, v)
    int _drag = -1;                 // vertex of the current key being dragged
    float _zoom = 1.0f;
    float _pan[2] = {0, 0};

    GLuint _tex = 0;
    int _tex_slice = -1;
};

}  // namespace gui
