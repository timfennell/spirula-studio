// TrainCapturePanel.cpp -- the trainer screen's "Training animation capture"
// section (GuiApp::draw_capture_panel); the capture itself is TrainCapture.

#include "app/gui/GuiApp.h"
#include "app/gui/Layout.h"
#include "app/gui/Ui.h"

#include <algorithm>
#include <cctype>
#include <filesystem>

namespace fs = std::filesystem;

namespace gui {

namespace {

const ImVec4 kWarn(0.95f, 0.75f, 0.30f, 1.0f);

bool has(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}
void toggle(std::vector<std::string>& v, const std::string& s, bool on) {
    v.erase(std::remove(v.begin(), v.end(), s), v.end());
    if (on) v.push_back(s);
}
bool icontains(const std::string& hay, const std::string& needle) {
    if (needle.empty()) return true;
    auto it = std::search(hay.begin(), hay.end(), needle.begin(), needle.end(),
                          [](char a, char b) { return std::tolower((unsigned char)a) ==
                                                      std::tolower((unsigned char)b); });
    return it != hay.end();
}

}  // namespace

void GuiApp::draw_capture_panel() {
    if (!ui::CollapsingHeaderRaw("Training animation capture###capture_header")) return;
    const TrainRunner::Phase ph = _runner.phase();
    const bool running = ph == TrainRunner::Phase::Preparing ||
                         ph == TrainRunner::Phase::Training;
    CaptureSettings& c = _capture;
    bool dirty = false;

    ImGui::BeginDisabled(running);
    dirty |= ui::CheckboxRaw("Capture image sequences while training###cap_on", &c.enabled);
    ImGui::BeginDisabled(!c.enabled);

    const char* res[] = {"HD  1920 x 1080", "2K  2560 x 1440", "4K  3840 x 2160"};
    ImGui::SetNextItemWidth(px(170.0f));
    dirty |= ui::ComboRaw("Resolution###cap_res", &c.resolution, res, 3);
    const char* outs[] = {"PNG frames", "Video (MP4)", "PNG frames + video"};
    ImGui::SetNextItemWidth(px(170.0f));
    dirty |= ui::ComboRaw("Output###cap_output", &c.output, outs, 3);
    if (c.output != 0) {
        const char* codecs[] = {"H.264", "H.265 (smaller)"};
        ImGui::SetNextItemWidth(px(170.0f));
        dirty |= ui::ComboRaw("Codec###cap_codec", &c.codec, codecs, 2);
        const char* quals[] = {"Best", "Balanced", "Smallest"};
        ImGui::SetNextItemWidth(px(170.0f));
        dirty |= ui::ComboRaw("Quality###cap_quality", &c.quality, quals, 3);
        ImGui::SetNextItemWidth(px(170.0f));
        if (ui::InputFloatRaw("Frames per second###cap_fps", &c.fps, "%.0f")) {
            c.fps = std::clamp(c.fps, 1.0f, 120.0f);
            dirty = true;
        }
    }
    if (c.output != 1) {
        int bg = c.transparent ? 0 : 1;
        const char* bgs[] = {"Transparent", "Black"};
        ImGui::SetNextItemWidth(px(170.0f));
        if (ui::ComboRaw("PNG background###cap_bg", &bg, bgs, 2)) { c.transparent = bg == 0; dirty = true; }
    }
    ImGui::SetNextItemWidth(px(170.0f));
    if (ui::InputIntRaw("Capture every N steps###cap_every", &c.every_steps)) {
        c.every_steps = std::max(1, c.every_steps);
        dirty = true;
    }
    dirty |= ui::CheckboxRaw("5x as often for the first 10 captures' worth of steps###cap_dense",
                             &c.dense_start);

    dirty |= ui::CheckboxRaw("Arc: move through the views ticked under Arc, in tick order###cap_arc",
                             &c.arc);
    dirty |= ui::CheckboxRaw("Orbit: circle the subject from the view chosen under Orbit###cap_orbit",
                             &c.orbit);
    if (c.orbit) {
        ImGui::SetNextItemWidth(px(170.0f));
        dirty |= ui::InputFloatRaw("Orbit degrees over the run###cap_deg", &c.orbit_degrees, "%.0f");
    }

    spirula::TrainerSession* s = _runner.session();
    const bool have = s && !s->ds.image_filenames.empty();
    if (!have) {
        ui::TextDisabledRaw("Open a dataset to choose views.");
    } else {
        ImGui::SetNextItemWidth(px(250.0f));
        ui::InputTextRaw("Filter views###cap_filter", &_capture_filter);
        const ImGuiTableFlags tf = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                                   ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingFixedFit;
        if (ImGui::BeginTable("##cap_views", 4, tf, ImVec2(0, px(220.0f)))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ui::TableSetupColumnRaw("View", ImGuiTableColumnFlags_WidthStretch);
            ui::TableSetupColumnRaw("Fixed", 0, px(44.0f));
            ui::TableSetupColumnRaw("Arc", 0, px(64.0f));
            ui::TableSetupColumnRaw("Orbit", 0, px(44.0f));
            ImGui::TableHeadersRow();
            const auto& names = s->ds.image_filenames;
            for (size_t i = 0; i < names.size(); i++) {
                const std::string name = fs::path(names[i]).filename().string();
                if (!icontains(name, _capture_filter)) continue;
                ImGui::PushID((int)i);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ui::TextRaw(name);
                ImGui::TableSetColumnIndex(1);
                bool fixed_on = has(c.fixed_views, name);
                if (ui::CheckboxRaw("##fixed", &fixed_on)) { toggle(c.fixed_views, name, fixed_on); dirty = true; }
                ImGui::TableSetColumnIndex(2);
                bool arc_on = has(c.arc_views, name);
                if (ui::CheckboxRaw("##arc", &arc_on)) { toggle(c.arc_views, name, arc_on); dirty = true; }
                if (arc_on) {
                    const auto at = std::find(c.arc_views.begin(), c.arc_views.end(), name);
                    ImGui::SameLine();
                    ui::TextDisabledRaw(std::to_string(at - c.arc_views.begin() + 1));
                }
                ImGui::TableSetColumnIndex(3);
                if (ui::RadioButtonRaw("##orbit", c.orbit_view == name)) {
                    c.orbit_view = c.orbit_view == name ? std::string() : name;
                    dirty = true;
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        if (!c.arc_views.empty() && ui::ButtonRaw("Clear arc###cap_clear_arc")) {
            c.arc_views.clear();
            dirty = true;
        }
    }

    const int frames = (int)capture_schedule(c, 0, std::max(1, _cfg.num_iterations)).size();
    int tracks = (int)c.fixed_views.size() + (c.arc && c.arc_views.size() >= 2 ? 1 : 0) +
                 (c.orbit && !c.orbit_view.empty() ? 1 : 0);
    int W = 0, H = 0;
    capture_frame_size(c.resolution, W, H);
    ui::TextDisabledRaw(std::to_string(tracks) + " sequence(s), about " + std::to_string(frames) +
                        " frames each at " + std::to_string(W) + " x " + std::to_string(H));
    ui::TextDisabledRaw("Written to <run folder>/captures/ (view_<name>, arc, orbit)");
    if (c.output != 0)
        ui::TextDisabledRaw("Video is encoded by ffmpeg on the CPU, over a black background.");
    if (c.arc && c.arc_views.size() < 2) ui::TextColoredRaw(kWarn, "The arc needs at least two views.");
    if (c.orbit && c.orbit_view.empty()) ui::TextColoredRaw(kWarn, "Choose the orbit's starting view.");

    ImGui::EndDisabled();
    ImGui::EndDisabled();
    if (dirty) save_settings();
}

}  // namespace gui
