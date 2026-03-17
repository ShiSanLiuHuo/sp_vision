/**
 * @brief 自动曝光调参工具
 *
 * 遍历 exposure_ms / gamma 参数组合，每组采样 N 帧，
 * 记录 YOLO 检测到的装甲板平均置信度，输出排名结果。
 *
 * 用法：
 *   ./build/tune_exposure configs/sentry_blue.yaml
 *
 * 可通过命令行参数调整搜索范围：
 *   --exp-min=0.5 --exp-max=5.0 --exp-step=0.5
 *   --gamma-min=0.6 --gamma-max=1.0 --gamma-step=0.1
 *   --frames=30   每组采样帧数
 */

#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

#include "io/camera.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h         |              | 输出帮助}"
  "{@config-path   | configs/sentry_blue.yaml | yaml配置文件路径}"
  "{exp-min        | 0.5          | 曝光最小值 ms}"
  "{exp-max        | 5.0          | 曝光最大值 ms}"
  "{exp-step       | 0.5          | 曝光步长 ms}"
  "{gamma-min      | 0.6          | gamma最小值}"
  "{gamma-max      | 1.0          | gamma最大值}"
  "{gamma-step     | 0.1          | gamma步长}"
  "{frames f       | 30           | 每组采样帧数}";

struct TuneResult
{
  double exposure_ms;
  double gamma;
  double avg_confidence;   // 所有检测到的装甲板置信度均值
  double detect_rate;      // 有检测结果的帧占比
  double score;            // avg_confidence * detect_rate，综合得分
};

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) { cli.printMessage(); return 0; }

  const std::string config_path = cli.get<std::string>(0);
  const double exp_min   = cli.get<double>("exp-min");
  const double exp_max   = cli.get<double>("exp-max");
  const double exp_step  = cli.get<double>("exp-step");
  const double gam_min   = cli.get<double>("gamma-min");
  const double gam_max   = cli.get<double>("gamma-max");
  const double gam_step  = cli.get<double>("gamma-step");
  const int    n_frames  = cli.get<int>("frames");

  // 读取基础 yaml
  YAML::Node base_yaml;
  try {
    base_yaml = YAML::LoadFile(config_path);
  } catch (const std::exception & e) {
    tools::logger()->error("Failed to load {}: {}", config_path, e.what());
    return 1;
  }

  const std::string tmp_yaml_path = "/tmp/_tune_exposure_tmp.yaml";

  // 构建参数网格
  std::vector<std::pair<double, double>> grid;
  for (double exp = exp_min; exp <= exp_max + 1e-6; exp += exp_step)
    for (double gam = gam_min; gam <= gam_max + 1e-6; gam += gam_step)
      grid.emplace_back(
        std::round(exp * 10) / 10.0,
        std::round(gam * 10) / 10.0);

  tools::logger()->info("共 {} 组参数，每组采样 {} 帧", grid.size(), n_frames);

  std::vector<TuneResult> results;
  tools::Exiter exiter;

  for (std::size_t idx = 0; idx < grid.size() && !exiter.exit(); ++idx) {
    auto [exp, gam] = grid[idx];

    // 写入临时 yaml
    YAML::Node tmp = YAML::Clone(base_yaml);
    tmp["exposure_ms"] = exp;
    tmp["gamma"]       = gam;
    {
      std::ofstream fout(tmp_yaml_path);
      fout << tmp;
    }

    tools::logger()->info("[{}/{}] exposure={:.1f}ms  gamma={:.1f}  采样中...",
                          idx + 1, grid.size(), exp, gam);

    double conf_sum   = 0.0;
    int    conf_count = 0;
    int    hit_frames = 0;

    try {
      io::Camera        camera(tmp_yaml_path);
      auto_aim::YOLO    yolo(config_path, false);   // debug=false 不弹窗

      // 预热：丢弃前几帧（相机刚开启图像可能不稳定）
      for (int w = 0; w < 5; ++w) {
        cv::Mat dummy; std::chrono::steady_clock::time_point ts;
        camera.read(dummy, ts);
      }

      for (int f = 0; f < n_frames && !exiter.exit(); ++f) {
        cv::Mat img; std::chrono::steady_clock::time_point ts;
        camera.read(img, ts);
        if (img.empty()) break;

        auto armors = yolo.detect(img);
        if (!armors.empty()) {
          ++hit_frames;
          for (const auto & a : armors) {
            conf_sum += a.confidence;
            ++conf_count;
          }
        }
      }
      // camera / yolo 析构，相机关闭
    } catch (const std::exception & e) {
      tools::logger()->warn("  跳过（相机初始化失败）: {}", e.what());
      // 相机重新上电需要一点时间
      std::this_thread::sleep_for(500ms);
      continue;
    }

    double avg_conf    = conf_count > 0 ? conf_sum / conf_count : 0.0;
    double detect_rate = static_cast<double>(hit_frames) / n_frames;
    double score       = avg_conf * detect_rate;

    tools::logger()->info("  avg_conf={:.4f}  detect_rate={:.2f}  score={:.4f}",
                          avg_conf, detect_rate, score);

    results.push_back({exp, gam, avg_conf, detect_rate, score});

    // 相机断开后稍作等待再重新打开，避免 USB 枚举问题
    std::this_thread::sleep_for(300ms);
  }

  // 清理临时文件
  std::filesystem::remove(tmp_yaml_path);

  if (results.empty()) {
    tools::logger()->warn("无有效结果");
    return 1;
  }

  // 按综合得分降序排序
  std::sort(results.begin(), results.end(),
            [](const TuneResult & a, const TuneResult & b) { return a.score > b.score; });

  fmt::print("\n========== 调参结果排名（Top 10）==========\n");
  fmt::print("{:>4}  {:>10}  {:>6}  {:>10}  {:>12}  {:>8}\n",
             "Rank", "exposure_ms", "gamma", "avg_conf", "detect_rate", "score");
  fmt::print("{}\n", std::string(60, '-'));
  for (std::size_t i = 0; i < std::min<std::size_t>(10, results.size()); ++i) {
    const auto & r = results[i];
    fmt::print("{:>4}  {:>10.1f}  {:>6.1f}  {:>10.4f}  {:>12.2f}  {:>8.4f}\n",
               i + 1, r.exposure_ms, r.gamma, r.avg_confidence, r.detect_rate, r.score);
  }

  const auto & best = results.front();
  fmt::print("\n>>> 最优参数：exposure_ms={:.1f}  gamma={:.1f}  (score={:.4f})\n\n",
             best.exposure_ms, best.gamma, best.score);

  // 提示是否写入原始 yaml
  fmt::print("是否将最优参数写入 {}？(y/n): ", config_path);
  char c;
  std::cin >> c;
  if (c == 'y' || c == 'Y') {
    YAML::Node out = YAML::LoadFile(config_path);
    out["exposure_ms"] = best.exposure_ms;
    out["gamma"]       = best.gamma;
    std::ofstream fout(config_path);
    fout << out;
    tools::logger()->info("已写入 exposure_ms={:.1f}  gamma={:.1f}", best.exposure_ms, best.gamma);
  }

  return 0;
}
