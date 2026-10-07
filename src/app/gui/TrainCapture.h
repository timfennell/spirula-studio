#pragma once

// TrainCapture -- PNG image sequences of the model as it trains, rendered on
// the training thread between steps from fixed training views, an arc through
// several views, or an orbit about the scene. Independent of every preview.
// Frames go to <run>/captures/<track>/: frame_%05d.png and/or <track>.mp4
// (ffmpeg, H.264 or H.265), with frames.csv mapping frames to steps.

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace spirula { class TrainerSession; }
namespace gui::render { class FrameSink; }

namespace gui {

struct CaptureSettings {
    bool enabled = false;
    int resolution = 0;          // 0 HD 1920x1080, 1 2K 2560x1440, 2 4K 3840x2160
    int every_steps = 100;
    bool dense_start = true;     // 5x more often over the first 10 intervals
    bool transparent = true;     // RGBA over nothing, else RGB over black
    int output = 1;              // 0 PNG frames, 1 video, 2 both
    int codec = 1;               // 0 H.264, 1 H.265 (video is always over black)
    int quality = 1;             // 0 best, 1 balanced, 2 smallest
    float fps = 30.0f;
    std::string ffmpeg = "ffmpeg";   // set from the app's tool path, not saved
    // Image file names (not indices) so a saved selection survives a re-parse.
    std::vector<std::string> fixed_views;
    bool arc = false;
    std::vector<std::string> arc_views;   // keyframes, in order
    bool orbit = false;
    std::string orbit_view;
    float orbit_degrees = 360.0f;

    bool any_track() const {
        return !fixed_views.empty() || (arc && arc_views.size() >= 2) ||
               (orbit && !orbit_view.empty());
    }
};

void capture_frame_size(int resolution, int& W, int& H);
// The steps a run from `start` to `total` captures at, ascending.
std::vector<int> capture_schedule(const CaptureSettings& cs, int start, int total);
// gui.conf persistence: capture_* keys.
void read_capture_setting(CaptureSettings& cs, const std::string& k, const std::string& v);
void write_capture_settings(FILE* f, const CaptureSettings& cs);

class TrainCapture {
public:
    ~TrainCapture() { finish(); }

    // Training thread, after setup_engine(). Logs what it will write.
    void begin(spirula::TrainerSession& s, const CaptureSettings& cs,
               int start_step, int total_steps);
    // Training thread, engine mutex NOT held: renders when `step` is due.
    void on_step(spirula::TrainerSession& s, int step);
    // Waits for the PNG writers to drain.
    void finish();

private:
    struct Pose { float R[9]; float C[3]; float intr[4]; int w = 0, h = 0;
                  std::string model, distortion; float dist[8]; };
    struct Track { std::string dir; std::vector<int> posts; int kind = 0;
                   std::shared_ptr<render::FrameSink> video; };
    struct Job { std::string path; int W = 0, H = 0, ch = 4;
                 std::vector<uint8_t> px, rgb; };   // rgb: over black, for video

    Pose post_pose(const spirula::TrainerSession& s, int p) const;
    Pose track_pose(const spirula::TrainerSession& s, const Track& t, float u) const;
    void render(spirula::TrainerSession& s, const Pose& p, Job& job);
    spirula::TrainerSession* _session = nullptr;   // for finish()'s log
    void writer_loop();

    CaptureSettings _cs;
    std::vector<Track> _tracks;
    std::vector<int> _steps;      // capture schedule, ascending
    size_t _next = 0;             // index into _steps
    int _W = 0, _H = 0;
    float _center[3] = {0, 0, 0}, _up[3] = {0, 0, 1};
    std::vector<float> _rgb, _ts; // host scratch, reused

    std::mutex _mu;
    std::condition_variable _cv_job, _cv_room;
    std::deque<Job> _queue;
    bool _closing = false;
    std::vector<std::thread> _writers;
};

}  // namespace gui
