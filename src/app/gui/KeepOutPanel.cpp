// KeepOutPanel.cpp -- the trainer screen's "Keep-out regions (insects)"
// section (GuiApp::draw_keep_out_panel): the cross-section void editor, which
// region to keep splats out of, whether its voids also block light, and
// switching it on for the next run or a paused one.

#include "app/gui/GuiApp.h"
#include "app/gui/Layout.h"
#include "app/gui/Ui.h"
#include "checkpoint/SplatPly.h"
#include "data/RoiDocument.h"
#include "i18n/catalog/TrainFields.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

namespace gui {

namespace {

const ImVec4 kWarn(0.95f, 0.75f, 0.30f, 1.0f);
const ImVec4 kOk(0.45f, 0.85f, 0.45f, 1.0f);
constexpr float kKeepOutDecay = 0.7f;
constexpr float kKeepOutWeight = 1e-4f;

std::string dataset_dir(const std::string& data) {
    fs::path p(data);
    if (p.filename() == "images") p = p.parent_path();
    return p.string();
}

// Densification is when the keep-out acts, so it has to run nearly to the end.
int keep_out_refine_stop(const TrainConfig& c) {
    return std::max(c.refine_stop_iter, (int)(0.875 * c.num_iterations));
}

// The newest run folder under <dataset>/outputs that holds a checkpoint.
std::string latest_run(const std::string& ds) {
    std::error_code ec;
    fs::path best;
    fs::file_time_type when{};
    for (const auto& e : fs::directory_iterator(fs::path(ds) / "outputs", ec)) {
        if (!e.is_directory(ec) || e.path().filename().string().rfind('.', 0) == 0) continue;
        try { (void)spirula::find_splat_ply(e.path().string()); } catch (...) { continue; }
        const auto t = fs::last_write_time(e.path(), ec);
        if (best.empty() || t > when) { best = e.path(); when = t; }
    }
    return best.string();
}

}  // namespace

std::string GuiApp::apply_keep_out_now(const std::string& region_file) {
    spirula::TrainerSession* s = _runner.session();
    if (!s || _runner.phase() != TrainRunner::Phase::Training || !_runner.paused())
        return "Pause training first.";
    try {
        std::lock_guard<std::mutex> lk(s->engine_mutex);
        s->cfg.roi_region = region_file;
        s->cfg.roi_outside_weight = kKeepOutWeight;
        s->cfg.roi_outside_opacity_decay = kKeepOutDecay;
        s->cfg.refine_stop_iter = keep_out_refine_stop(s->cfg);
        s->cfg.roi_void_stop = _keepout_block_light;
        s->roi.reset();
        s->load_region();
        s->setup_region();
    } catch (const std::exception& e) {
        return std::string("Could not apply: ") + e.what();
    }
    return "Applied " + fs::path(region_file).filename().string() +
           ": resume, and splats inside the voids fade out.";
}

void GuiApp::draw_keep_out_panel() {
    // The editor is a window of its own; it asks, the app applies.
    {
        const bool paused = _runner.phase() == TrainRunner::Phase::Training && _runner.paused();
        _void_editor.set_can_apply(paused, paused ? "" : "Pause training to apply the voids to the run.");
        const std::string req = _void_editor.take_apply_request();
        if (!req.empty()) {
            _keepout_file = req;
            _void_editor.set_note(apply_keep_out_now(req));
        }
    }
    if (!ui::CollapsingHeaderRaw("Keep-out regions (insects)###keepout_header")) return;
    ui::TextWrappedRaw("Keep splats out of voids inside the subject, so reflections on shiny "
                       "bodies cannot be built as structure inside them and view-dependent "
                       "colour stays on the surface. Train briefly, pause, outline the voids "
                       "on cross sections of the model, apply, and resume.");

    spirula::TrainerSession* s = _runner.session();
    const bool have = s && !s->ds.image_filenames.empty();
    const std::string dir = have ? dataset_dir(_cfg.data) : std::string();
    const TrainRunner::Phase ph = _runner.phase();
    const bool running = ph == TrainRunner::Phase::Training;
    const bool paused = running && _runner.paused();

    // ---- the void editor ----
    ImGui::BeginDisabled(!have || (running && !paused) || ph == TrainRunner::Phase::Preparing);
    if (ui::ButtonRaw("Cross-section void editor...###void_open")) {
        std::string ply;
        if (paused) {
            // The model as it is now, written where the run keeps its checkpoints.
            std::lock_guard<std::mutex> lk(s->engine_mutex);
            s->save_checkpoint(s->cur_step.load(),
                               s->cfg.save_full_checkpoint || s->cfg.save_only_latest_checkpoint);
            ply = s->out_dir.string();
        } else if (ph == TrainRunner::Phase::Done) {
            ply = s->out_dir.string();
        } else {
            ply = latest_run(dir);
        }
        const double rs = _cfg.relative_scale.value_or(1.0f);
        double focus[3];
        cameras_focus(s->ds, rs, s->ds.center.data(), focus);
        _void_editor.open(dir, ply, rs, s->ds.center.data(), focus);
    }
    ImGui::EndDisabled();
    if (running && !paused) ui::TextDisabledRaw("Pause training to slice the model as it is now.");
    else if (!running && have) ui::TextDisabledRaw("Not training: slices the newest saved run.");

    // ---- which region, and switching it on ----
    std::vector<std::string> files = have ? spirula::list_roi_files(dir) : std::vector<std::string>{};
    if (!files.empty()) {
        if (std::find(files.begin(), files.end(), _keepout_file) == files.end()) {
            _keepout_file = files.front();
            for (const std::string& f : files)
                if (fs::path(f).filename() == "voids.json") _keepout_file = f;
        }
        ImGui::SetNextItemWidth(px(220.0f));
        if (ui::BeginComboRaw("Keep-out region###keepout_file",
                              fs::path(_keepout_file).stem().string().c_str())) {
            for (const std::string& f : files)
                if (ui::SelectableRaw(fs::path(f).stem().string(), f == _keepout_file)) _keepout_file = f;
            ImGui::EndCombo();
        }
        ui::Checkbox(spirula::i18n::msg::field::roi_void_stop, &_keepout_block_light);
        if (ImGui::IsItemHovered()) ui::SetTooltipWrapped(spirula::i18n::msg::field::roi_void_stop_help);
        ImGui::BeginDisabled(running || ph == TrainRunner::Phase::Preparing);
        if (ui::ButtonRaw("Use for the next run###keepout_use")) {
            _cfg.roi_region = fs::path(_keepout_file).lexically_relative(dir).generic_string();
            _cfg.roi_outside_weight = kKeepOutWeight;
            _cfg.roi_outside_opacity_decay = kKeepOutDecay;
            _cfg.roi_mask_pixels = false;
            _cfg.refine_stop_iter = keep_out_refine_stop(_cfg);
            _cfg.roi_void_stop = _keepout_block_light;
            for (const char* k : {"roi_region", "roi_outside_weight", "roi_outside_opacity_decay",
                                  "roi_mask_pixels", "refine_stop_iter", "roi_void_stop"})
                _cfg_ui.touched.insert(k);
            _parse_dirty = true;
            _keepout_note = "The next run keeps splats out of " + fs::path(_keepout_file).stem().string() + ".";
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(!paused);
        if (ui::ButtonRaw("Apply to this run now###keepout_apply")) _keepout_note = apply_keep_out_now(_keepout_file);
        ImGui::EndDisabled();
    } else if (have) {
        ui::TextDisabledRaw("No keep-out regions in this dataset's roi folder yet.");
    }
    if (!_keepout_note.empty()) ui::TextWrappedRaw(_keepout_note);
}

}  // namespace gui
