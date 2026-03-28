#include "io/camera.hpp"

#include <opencv2/opencv.hpp>
#include <filesystem>

#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

int main(int argc, char * argv[])
{
  // Simple, robust command-line parsing so flags like -d won't be misinterpreted
  std::string config_path = "configs/camera.yaml";
  std::string output_path = "records/output.mp4";
  bool display = false;
  double duration = -1.0;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help" || a == "-?") {
      std::cout << "Usage: " << argv[0] << " [config-path] [-o output] [-t duration] [-d]\n";
      return 0;
    }
    if (a == "-o" || a == "--output") {
      if (i + 1 < argc) { output_path = argv[++i]; continue; }
      std::cerr << "-o requires an argument\n";
      return 1;
    }
    if (a == "-t" || a == "--duration") {
      if (i + 1 < argc) { duration = std::atof(argv[++i]); continue; }
      std::cerr << "-t requires an argument\n";
      return 1;
    }
    if (a == "-d" || a == "--display") {
      display = true;
      continue;
    }

    // positional first non-option arg is config path
    if (a.size() > 0 && a[0] != '-') {
      if (config_path == "configs/camera.yaml") config_path = a;
      else {
        // ignore extra positional
      }
      continue;
    }
  }

  tools::Exiter exiter;

  // ensure output directory exists
  try {
    std::filesystem::path p(output_path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
  } catch (...) {
    tools::logger()->warn("Failed to create output directory for {}", output_path);
  }

  io::Camera camera(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  cv::VideoWriter writer;
  bool writer_initialized = false;
  auto start_time = std::chrono::steady_clock::now();
  double last_dt = 0.0;

  // Warm-up: buffer several frames and use their timestamps to estimate actual camera FPS
  const int warmup_frames = 10;
  std::vector<cv::Mat> frame_buffer;
  std::vector<std::chrono::steady_clock::time_point> ts_buffer;

  tools::logger()->info("Recording to {} (display={})", output_path, display);

  while (!exiter.exit()) {
    camera.read(img, timestamp);
    if (img.empty()) {
      tools::logger()->warn("Empty frame, skipping...");
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }

    // warm-up buffering for FPS estimation
    if (!writer_initialized) {
      frame_buffer.push_back(img.clone());
      ts_buffer.push_back(timestamp);

      // if we have enough timestamps, estimate fps
      if (ts_buffer.size() >= 2) {
        if (static_cast<int>(ts_buffer.size()) >= warmup_frames) {
          // compute average dt
          double sum_dt = 0.0;
          int n = static_cast<int>(ts_buffer.size());
          for (int i = 0; i < n - 1; ++i) {
            sum_dt += tools::delta_time(ts_buffer[i + 1], ts_buffer[i]);
          }
          double avg_dt = sum_dt / (n - 1);
          double fps = avg_dt > 1e-6 ? 1.0 / avg_dt : 30.0;
          // clamp reasonable fps
          fps = std::clamp(fps, 1.0, 120.0);
          // Use an integer FPS to avoid FFmpeg timebase/denominator issues
          int fps_used = static_cast<int>(std::round(fps));

          int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
          cv::Size size(img.cols, img.rows);
          writer.open(output_path, fourcc, fps_used, size, true);
          if (!writer.isOpened()) {
            tools::logger()->error("Failed to open VideoWriter {} (tried fps={:.2f})", output_path, fps);
            return 1;
          }
          writer_initialized = true;
          tools::logger()->info("VideoWriter initialized ({}x{}, {:.0f} fps) — measured {:.2f} fps, using {} fps",
                               size.width, size.height, static_cast<double>(fps_used), fps, fps_used);

          // flush buffered frames into writer
          for (const auto & f : frame_buffer) writer.write(f);
          frame_buffer.clear();
        }
        else {
          // not enough frames yet: wait for more
          continue;
        }
      } else {
        continue; // wait for next frame
      }
    }

    // after initialization, just write frames
    writer.write(img);

    // display and fps logging
    static std::chrono::steady_clock::time_point last_stamp = timestamp;
    auto dt = tools::delta_time(timestamp, last_stamp);
    last_stamp = timestamp;
    if (dt > 1e-6) last_dt = dt;
    tools::logger()->debug("frame dt={:.3f} ms", last_dt * 1e3);

    if (display) {
      cv::imshow("record", img);
      int key = cv::waitKey(1);
      if (key == 'q') break;
    }

    if (duration > 0) {
      auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
      if (elapsed >= duration) break;
    }
  }

  if (writer.isOpened()) writer.release();
  tools::logger()->info("Recording finished");

  return 0;
}