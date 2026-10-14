// VoidEditor.cpp -- see VoidEditor.h.

#include "app/gui/VoidEditor.h"

#include "app/gui/Layout.h"
#include "app/gui/Ui.h"
#include "checkpoint/SplatPly.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

namespace gui {

namespace {

const ImU32 kBlobColours[] = {IM_COL32(255, 90, 90, 255),  IM_COL32(90, 200, 255, 255),
                              IM_COL32(120, 230, 110, 255), IM_COL32(255, 200, 70, 255),
                              IM_COL32(220, 120, 255, 255), IM_COL32(255, 150, 200, 255)};
ImU32 blob_colour(int i, int alpha) {
    const ImU32 c = kBlobColours[i % 6];
    return (c & 0x00FFFFFF) | ((ImU32)alpha << 24);
}
const ImVec4 kWarn(0.95f, 0.75f, 0.30f, 1.0f);

std::string drawing_path(const std::string& ds) { return (fs::path(ds) / "roi" / "voids.svoid").string(); }
std::string region_path(const std::string& ds) { return (fs::path(ds) / "roi" / "voids.json").string(); }

}  // namespace

VoidEditor::~VoidEditor() {
    if (_gen.valid()) _gen.wait();
}

void VoidEditor::destroy_gl() {
    if (_tex) glDeleteTextures(1, &_tex);
    _tex = 0;
    _tex_slice = -1;
}

void VoidEditor::open(const std::string& dataset_dir, const std::string& ply, double rs,
                      const double center[3], const double focus[3]) {
    std::copy(focus, focus + 3, _focus);
    _open = true;
    _dataset = dataset_dir;
    _rs = rs > 0 ? rs : 1;
    std::copy(center, center + 3, _center);
    _error.clear();
    _note.clear();
    VoidDoc saved;
    std::string err;
    const bool have_saved = load_void_doc(drawing_path(dataset_dir), saved, err);
    _ply = ply.empty() && have_saved ? saved.source : ply;
    if (_ply.empty()) { _error = "There is no model to slice yet: train for a while first."; return; }
    if (have_saved) {
        // Same model frame as before, so the blobs stay where they were drawn.
        _doc = saved;
        _axis = saved.f.axis;
        _slices = saved.f.slices;
        start(false);
    } else {
        _doc = VoidDoc{};
        start(true);
    }
}

void VoidEditor::start(bool new_frame) {
    if (_gen.valid()) return;
    _error.clear();
    VoidDoc doc = _doc;
    const std::string ply = _ply;
    const double rs = _rs;
    const double c[3] = {_center[0], _center[1], _center[2]};
    const double fo[3] = {_focus[0], _focus[1], _focus[2]};
    const int axis = _axis, slices = _slices, res = _res;
    _gen = std::async(std::launch::async, [=]() mutable {
        Gen g;
        try {
            const std::string path = spirula::find_splat_ply(ply).first;
            const spirula::SplatCloud cloud = spirula::read_splat_ply(path, false);
            // Recutting along the same axis keeps the frame, so the blobs stay
            // where they were drawn; another axis starts over.
            if (new_frame && !doc.blobs.empty() && doc.f.slices > 0 && doc.f.axis == axis) {
                restack_blobs(doc, refine_void_frame(doc.f, slices, res));
            } else if (new_frame) {
                doc.f = make_void_frame(cloud, rs, c, fo, axis, slices, res);
                doc.blobs.clear();
            }
            doc.source = path;
            g.imgs = render_void_slices(cloud, rs, c, doc.f);
            g.doc = std::move(doc);
        } catch (const std::exception& e) {
            g.error = e.what();
        }
        return g;
    });
}

std::string VoidEditor::take_apply_request() {
    std::string r;
    r.swap(_apply_req);
    return r;
}

void VoidEditor::save() {
    std::error_code ec;
    fs::create_directories(fs::path(_dataset) / "roi", ec);
    save_void_doc(_doc, drawing_path(_dataset));
    const std::string region = void_region_json(_doc);
    if (region.empty()) {
        fs::remove(region_path(_dataset), ec);
        _note = "Saved the drawing; no blobs yet, so no region.";
    } else {
        FILE* f = std::fopen(region_path(_dataset).c_str(), "wb");
        if (!f) { _error = "Cannot write " + region_path(_dataset); return; }
        std::fwrite(region.data(), 1, region.size(), f);
        std::fclose(f);
        _note = "Saved " + region_path(_dataset);
    }
    _dirty = false;
}

void VoidEditor::draw() {
    if (!_open) return;
    if (_gen.valid() && _gen.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        Gen g = _gen.get();
        if (!g.error.empty()) {
            _error = g.error;
        } else {
            _doc = std::move(g.doc);
            _imgs = std::move(g.imgs);
            _slice = std::clamp(_slice, 0, std::max(0, _doc.f.slices - 1));
            if (_blob >= (int)_doc.blobs.size()) _blob = (int)_doc.blobs.size() - 1;
            _tex_slice = -1;
        }
    }
    ImGui::SetNextWindowSize(ImVec2(px(1200.0f), px(820.0f)), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Cross-section void editor###void_editor", &_open)) {
        ImGui::BeginChild("##void_controls", ImVec2(px(340.0f), 0), ImGuiChildFlags_Borders);
        draw_controls();
        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginChild("##void_canvas_child", ImVec2(0, 0), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        draw_canvas();
        ImGui::EndChild();
    }
    ImGui::End();
}

void VoidEditor::draw_controls() {
    const bool busy = _gen.valid();
    ui::TextWrappedRaw("Model: " + (_ply.empty() ? std::string("(none)") : _ply));
    if (busy) ui::TextColoredRaw(kWarn, "Making cross sections...");

    const char* axes[] = {"Across the body's long axis", "Across its middle axis", "Across its short axis",
                          "Along dataset X", "Along dataset Y", "Along dataset Z"};
    ImGui::SetNextItemWidth(-1);
    ui::ComboRaw("##void_axis", &_axis, axes, 6);
    ImGui::SetNextItemWidth(-1);
    ui::SliderIntRaw("##void_slices", &_slices, 32, 256, "%d slices");
    ImGui::SetNextItemWidth(-1);
    ui::SliderIntRaw("##void_res", &_res, 256, 1024, "%d px images");
    ImGui::BeginDisabled(busy || _ply.empty());
    if (ui::ButtonRaw("Make new cross sections###void_regen")) start(true);
    if (!_doc.blobs.empty())
        ui::TextDisabledRaw(_axis == _doc.f.axis ? "Same axis: the blobs are kept and restacked."
                                                 : "Another axis: the blobs are cleared.");
    ImGui::EndDisabled();
    ImGui::Separator();

    const int n = _doc.f.slices;
    if (n > 0 && !_imgs.empty()) {
        if (ui::ArrowButtonRaw("##void_prev", ImGuiDir_Left) && _slice > 0) { _slice--; _drawing.clear(); }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(px(230.0f));
        if (ui::SliderIntRaw("##void_slice", &_slice, 0, n - 1, "slice %d")) _drawing.clear();
        ImGui::SameLine();
        if (ui::ArrowButtonRaw("##void_next", ImGuiDir_Right) && _slice < n - 1) { _slice++; _drawing.clear(); }
        ui::TextDisabledRaw("Up/down arrow keys step slices too.");
    }
    ImGui::Separator();

    ui::TextRaw("Void blobs");
    for (int i = 0; i < (int)_doc.blobs.size(); i++) {
        ImGui::PushID(i);
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float h = ImGui::GetTextLineHeight();
        ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x + h, p.y + h), blob_colour(i, 255));
        ImGui::Dummy(ImVec2(h, h));
        ImGui::SameLine();
        const std::string label = _doc.blobs[(size_t)i].name + "  (" +
                                  std::to_string(_doc.blobs[(size_t)i].keys.size()) + " key slices)";
        if (ui::SelectableRaw(label, _blob == i)) { _blob = i; _drawing.clear(); }
        ImGui::PopID();
    }
    ImGui::BeginDisabled(_imgs.empty());
    if (ui::ButtonRaw("New blob###void_new")) {
        _doc.blobs.push_back({"Void " + std::to_string(_doc.blobs.size() + 1), {}});
        _blob = (int)_doc.blobs.size() - 1;
        _drawing.clear();
        _dirty = true;
    }
    ImGui::EndDisabled();
    if (_blob >= 0 && _blob < (int)_doc.blobs.size()) {
        ImGui::SameLine();
        if (ui::ButtonRaw("Delete blob###void_del")) {
            _doc.blobs.erase(_doc.blobs.begin() + _blob);
            _blob = std::min(_blob, (int)_doc.blobs.size() - 1);
            _dirty = true;
        }
    }
    if (_blob >= 0 && _blob < (int)_doc.blobs.size()) {
        VoidBlob& b = _doc.blobs[(size_t)_blob];
        ImGui::SetNextItemWidth(-1);
        if (ui::InputTextRaw("##void_name", &b.name)) _dirty = true;
        if (!b.keys.empty()) {
            int first, last;
            blob_span(b, _doc.f.pitch, first, last);
            ui::TextDisabledRaw("Spans slices " + std::to_string(std::max(0, first)) + " to " +
                                std::to_string(std::min(_doc.f.slices - 1, last)) +
                                (b.round_ends ? " (ends rounded)" : " (ends at the first and last key)"));
        }
        if (ui::CheckboxRaw("Round the ends past the first and last key###void_round", &b.round_ends))
            _dirty = true;
        bool is_key = false;
        const std::vector<VPt> here = blob_outline(b, _slice, _doc.f.pitch, &is_key);
        ui::TextDisabledRaw(is_key ? "This slice: key outline (drag its points)"
                            : here.empty() ? "This slice: blob not here"
                                           : "This slice: interpolated");
        if (is_key && ui::ButtonRaw("Remove outline from this slice###void_clear")) {
            set_key(b, _slice, {});
            _dirty = true;
        }
        if (!is_key && !here.empty() && ui::ButtonRaw("Make this outline a key###void_promote")) {
            std::vector<VPt> pts;
            for (size_t i = 0; i < here.size(); i += std::max<size_t>(1, here.size() / 12)) pts.push_back(here[i]);
            set_key(b, _slice, pts);
            _dirty = true;
        }
        if (!b.keys.empty()) {
            ui::TextDisabledRaw("Key slices (click to go):");
            for (size_t k = 0; k < b.keys.size(); k++) {
                if (k % 6) ImGui::SameLine();
                ImGui::PushID((int)k);
                if (ui::ButtonRaw(std::to_string(b.keys[k].slice).c_str())) { _slice = b.keys[k].slice; _drawing.clear(); }
                ImGui::PopID();
            }
        }
    }
    ImGui::Separator();

    if (ui::ButtonRaw(_dirty ? "Save voids *###void_save" : "Save voids###void_save")) save();
    ImGui::SameLine();
    ImGui::BeginDisabled(!_can_apply);
    if (ui::ButtonRaw("Apply to paused run###void_apply")) {
        save();
        if (_error.empty()) _apply_req = region_path(_dataset);
    }
    ImGui::EndDisabled();
    if (!_can_apply && !_apply_why.empty()) ui::TextDisabledRaw(_apply_why);
    if (!_note.empty()) ui::TextWrappedRaw(_note);
    if (!_error.empty()) ui::TextColoredWrappedRaw(kWarn, _error);
    ImGui::Separator();
    ui::TextWrappedRaw("Drawing: pick or make a blob, then click around the void to place points. "
                       "Click the first point, double-click or press Enter to close it. "
                       "Right-click or Backspace removes the last point, Esc cancels. "
                       "Outline the blob on a few slices; the slices between are interpolated "
                       "and the ends rounded off. Wheel zooms, middle-drag pans.");
}

void VoidEditor::draw_canvas() {
    const VoidFrame& f = _doc.f;
    if (_imgs.empty() || f.W <= 0) {
        ui::TextDisabledRaw(_gen.valid() ? "Making cross sections..." : "No cross sections.");
        return;
    }
    if (_tex_slice != _slice && _slice < (int)_imgs.size()) {
        if (!_tex) {
            glGenTextures(1, &_tex);
            glBindTexture(GL_TEXTURE_2D, _tex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        }
        glBindTexture(GL_TEXTURE_2D, _tex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, f.W, f.H, 0, GL_RGB, GL_UNSIGNED_BYTE,
                     _imgs[(size_t)_slice].data());
        _tex_slice = _slice;
    }
    const ImVec2 c0 = ImGui::GetCursorScreenPos();
    ImVec2 avail = ImGui::GetContentRegionAvail();
    avail.x = std::max(avail.x, 64.0f);
    avail.y = std::max(avail.y, 64.0f);
    ui::InvisibleButtonRaw("##void_canvas", avail, ImGuiButtonFlags_MouseButtonLeft |
                                                       ImGuiButtonFlags_MouseButtonRight |
                                                       ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered = ImGui::IsItemHovered();
    ImGuiIO& io = ImGui::GetIO();
    const float base = std::min(avail.x / f.W, avail.y / f.H);
    float scale = base * _zoom;
    ImVec2 org(c0.x + (avail.x - f.W * scale) * 0.5f + _pan[0], c0.y + (avail.y - f.H * scale) * 0.5f + _pan[1]);
    auto to_screen = [&](const VPt& p) {
        return ImVec2(org.x + (float)((p[0] - f.u0) / f.px) * scale, org.y + (float)((p[1] - f.v0) / f.px) * scale);
    };
    auto to_uv = [&](ImVec2 s) -> VPt {
        return {f.u0 + (s.x - org.x) / scale * f.px, f.v0 + (s.y - org.y) / scale * f.px};
    };

    if (hovered && io.MouseWheel != 0) {
        const VPt at = to_uv(io.MousePos);
        _zoom = std::clamp(_zoom * std::pow(1.15f, io.MouseWheel), 0.5f, 40.0f);
        scale = base * _zoom;
        org = ImVec2(c0.x + (avail.x - f.W * scale) * 0.5f + _pan[0], c0.y + (avail.y - f.H * scale) * 0.5f + _pan[1]);
        const ImVec2 now = to_screen(at);
        _pan[0] += io.MousePos.x - now.x;
        _pan[1] += io.MousePos.y - now.y;
        org = ImVec2(c0.x + (avail.x - f.W * scale) * 0.5f + _pan[0], c0.y + (avail.y - f.H * scale) * 0.5f + _pan[1]);
    }
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
        _pan[0] += io.MouseDelta.x;
        _pan[1] += io.MouseDelta.y;
    }
    if (hovered || ImGui::IsWindowFocused()) {
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow) && _slice < f.slices - 1) { _slice++; _drawing.clear(); }
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow) && _slice > 0) { _slice--; _drawing.clear(); }
    }

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->PushClipRect(c0, ImVec2(c0.x + avail.x, c0.y + avail.y), true);
    dl->AddImage((ImTextureID)(intptr_t)_tex, org, ImVec2(org.x + f.W * scale, org.y + f.H * scale));

    for (int i = 0; i < (int)_doc.blobs.size(); i++) {
        bool is_key = false;
        const std::vector<VPt> o = blob_outline(_doc.blobs[(size_t)i], _slice, f.pitch, &is_key);
        if (o.size() < 3) continue;
        std::vector<ImVec2> pts;
        for (const VPt& p : o) pts.push_back(to_screen(p));
        const bool sel = i == _blob;
        dl->AddPolyline(pts.data(), (int)pts.size(), blob_colour(i, sel ? 255 : 150), ImDrawFlags_Closed,
                        is_key ? 2.5f : 1.2f);
    }

    VoidBlob* b = (_blob >= 0 && _blob < (int)_doc.blobs.size()) ? &_doc.blobs[(size_t)_blob] : nullptr;
    VoidKey* key = nullptr;
    if (b)
        for (VoidKey& k : b->keys) if (k.slice == _slice) key = &k;
    const float grab = 8.0f;
    if (b && hovered) {
        auto close = [&]() {
            if (_drawing.size() >= 3) { set_key(*b, _slice, _drawing); _dirty = true; }
            _drawing.clear();
        };
        if (!_drawing.empty()) {
            const ImVec2 first = to_screen(_drawing.front());
            const bool near_first = std::hypot(io.MousePos.x - first.x, io.MousePos.y - first.y) < grab;
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) close();
            else if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                if (near_first && _drawing.size() >= 3) close();
                else _drawing.push_back(to_uv(io.MousePos));
            }
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) || ImGui::IsKeyPressed(ImGuiKey_Backspace))
                if (!_drawing.empty()) _drawing.pop_back();
            if (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)) close();
            if (ImGui::IsKeyPressed(ImGuiKey_Escape)) _drawing.clear();
        } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            _drag = -1;
            if (key)
                for (size_t i = 0; i < key->pts.size(); i++) {
                    const ImVec2 s = to_screen(key->pts[i]);
                    if (std::hypot(io.MousePos.x - s.x, io.MousePos.y - s.y) < grab) { _drag = (int)i; break; }
                }
            if (_drag < 0) _drawing.push_back(to_uv(io.MousePos));
        }
    }
    if (_drag >= 0 && key && _drag < (int)key->pts.size()) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            key->pts[(size_t)_drag] = to_uv(io.MousePos);
            _dirty = true;
        } else {
            _drag = -1;
        }
    }
    if (key && b)
        for (const VPt& p : key->pts) {
            const ImVec2 s = to_screen(p);
            dl->AddRectFilled(ImVec2(s.x - 3, s.y - 3), ImVec2(s.x + 3, s.y + 3), blob_colour(_blob, 255));
        }
    if (!_drawing.empty()) {
        std::vector<ImVec2> pts;
        for (const VPt& p : _drawing) pts.push_back(to_screen(p));
        pts.push_back(io.MousePos);
        dl->AddPolyline(pts.data(), (int)pts.size(), IM_COL32(255, 255, 255, 230), 0, 1.5f);
        for (size_t i = 0; i + 1 < pts.size(); i++) dl->AddCircleFilled(pts[i], 3.5f, IM_COL32(255, 255, 255, 255));
    }
    char info[160];
    std::snprintf(info, sizeof info, "slice %d / %d    %s", _slice, f.slices - 1,
                  b ? b->name.c_str() : "no blob selected: make one to draw");
    dl->AddText(ImVec2(c0.x + 8, c0.y + 6), IM_COL32(230, 230, 230, 255), info);
    dl->PopClipRect();
}

}  // namespace gui
