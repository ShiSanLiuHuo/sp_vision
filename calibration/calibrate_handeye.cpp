#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <filesystem>
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明}"
  "{config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{input-folder i |                          | 输入文件夹路径   }";

struct Args
{
  std::string config_path = "configs/calibration.yaml";
  std::string input_folder;
};

static void print_usage(const char * argv0)
{
  fmt::print(
    "Usage:\n"
    "  {} -c <config.yaml> -i <folder>\n"
    "  {} --config-path=<config.yaml> --input-folder=<folder>\n"
    "  {} -c <config.yaml> <folder>\n",
    argv0, argv0, argv0);
}

static bool starts_with(const std::string & s, const std::string & prefix)
{
  return s.rfind(prefix, 0) == 0;
}

static Args parse_args(int argc, char * argv[])
{
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help" || a == "-?" || a == "--usage") {
      print_usage(argv[0]);
      std::exit(0);
    }

    auto take_value = [&](std::string & out) {
      if (i + 1 >= argc) {
        fmt::print(stderr, "Error: {} expects a value\n", a);
        print_usage(argv[0]);
        std::exit(1);
      }
      out = argv[++i];
    };

    if (a == "-c" || a == "--config-path") {
      take_value(args.config_path);
      continue;
    }
    if (starts_with(a, "-c=") || starts_with(a, "--config-path=")) {
      args.config_path = a.substr(a.find('=') + 1);
      continue;
    }

    if (a == "-i" || a == "--input-folder") {
      take_value(args.input_folder);
      continue;
    }
    if (starts_with(a, "-i=") || starts_with(a, "--input-folder=")) {
      args.input_folder = a.substr(a.find('=') + 1);
      continue;
    }

    if (!a.empty() && a[0] != '-' && args.input_folder.empty()) {
      args.input_folder = a;
      continue;
    }

    fmt::print(stderr, "Error: unknown argument: {}\n", a);
    print_usage(argv[0]);
    std::exit(1);
  }
  return args;
}

static cv::Ptr<cv::SimpleBlobDetector> make_blob_detector(int blob_color)
{
  cv::SimpleBlobDetector::Params p;
  p.filterByColor = true;
  p.blobColor = static_cast<uchar>(blob_color);

  // 这些阈值偏宽松，适配不同曝光/对比度
  p.minThreshold = 5;
  p.maxThreshold = 220;
  p.thresholdStep = 10;

  p.filterByArea = true;
  p.minArea = 30;
  p.maxArea = 500000;

  p.filterByCircularity = false;
  p.filterByConvexity = false;
  p.filterByInertia = false;

  return cv::SimpleBlobDetector::create(p);
}

static cv::Mat to_gray8(const cv::Mat & src)
{
  cv::Mat gray;
  if (src.channels() == 3)
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
  else if (src.channels() == 1)
    gray = src;
  else
    src.convertTo(gray, CV_8U);

  if (gray.depth() != CV_8U)
    gray.convertTo(gray, CV_8U);
  return gray;
}

static bool find_chessboard_robust(
  const cv::Mat & gray, const cv::Size & pattern_size, std::vector<cv::Point2f> & corners_2d)
{
  corners_2d.clear();
  bool ok = false;
  try {
    ok = cv::findChessboardCornersSB(gray, pattern_size, corners_2d);
  } catch (const cv::Exception &) {
    ok = false;
  }

  if (!ok) {
    try {
      ok = cv::findChessboardCorners(
        gray, pattern_size, corners_2d,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    } catch (const cv::Exception &) {
      ok = false;
    }
  }

  if (ok) {
    cv::cornerSubPix(
      gray, corners_2d, cv::Size(11, 11), cv::Size(-1, -1),
      cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.1));
  }
  return ok;
}

static bool find_circles_grid_robust(
  const cv::Mat & img, const cv::Size & pattern_size, std::vector<cv::Point2f> & centers_2d,
  int & used_flags)
{
  // 统一转成 8-bit 灰度，避免不同相机输出格式影响检测
  cv::Mat gray = to_gray8(img);

  struct Try {
    int flags;
    int blob_color;
  };

  // 依次尝试：对称/非对称 + clustering(可提升稳定性) + 黑/白点
  const int sym = cv::CALIB_CB_SYMMETRIC_GRID;
  const int asym = cv::CALIB_CB_ASYMMETRIC_GRID;
  const int cl = cv::CALIB_CB_CLUSTERING;
  const Try tries[] = {
    {sym | cl, 0},  {sym | cl, 255},
    {sym, 0},       {sym, 255},
    {asym | cl, 0}, {asym | cl, 255},
    {asym, 0},      {asym, 255},
  };

  for (const auto & t : tries) {
    centers_2d.clear();
    auto detector = make_blob_detector(t.blob_color);
    try {
      if (cv::findCirclesGrid(gray, pattern_size, centers_2d, t.flags, detector)) {
        used_flags = t.flags;
        return true;
      }
    } catch (const cv::Exception &) {
      // 忽略异常，继续尝试其他组合
    }
  }

  used_flags = 0;
  return false;
}

enum class PatternKind
{
  Chessboard,
  CirclesGrid
};

static bool detect_pattern_locked(
  const cv::Mat & img, const cv::Size & configured_size, bool & locked, PatternKind & locked_kind,
  cv::Size & locked_size, int & locked_circles_flags, std::vector<cv::Point2f> & points_2d,
  PatternKind & used_kind, cv::Size & used_size, int & used_circles_flags)
{
  auto gray = to_gray8(img);

  auto try_detect = [&](PatternKind k, const cv::Size & sz, std::vector<cv::Point2f> & out,
                        int & circles_flags) {
    circles_flags = 0;
    if (k == PatternKind::Chessboard) {
      return find_chessboard_robust(gray, sz, out);
    }
    return find_circles_grid_robust(gray, sz, out, circles_flags);
  };

  if (locked) {
    used_kind = locked_kind;
    used_size = locked_size;
    used_circles_flags = locked_circles_flags;
    int tmp_flags = used_circles_flags;
    bool ok = try_detect(used_kind, used_size, points_2d, tmp_flags);
    if (used_kind == PatternKind::CirclesGrid)
      used_circles_flags = tmp_flags;
    return ok;
  }

  // 未锁定时：按 (cols,rows) 与 (rows,cols) 都试一遍，棋盘格优先。
  const cv::Size candidates[] = {configured_size, cv::Size(configured_size.height, configured_size.width)};
  for (auto sz : candidates) {
    used_kind = PatternKind::Chessboard;
    int circles_flags = 0;
    if (try_detect(used_kind, sz, points_2d, circles_flags)) {
      used_size = sz;
      used_circles_flags = 0;
      locked = true;
      locked_kind = used_kind;
      locked_size = used_size;
      locked_circles_flags = 0;
      return true;
    }
  }

  for (auto sz : candidates) {
    used_kind = PatternKind::CirclesGrid;
    int circles_flags = 0;
    if (try_detect(used_kind, sz, points_2d, circles_flags)) {
      used_size = sz;
      used_circles_flags = circles_flags;
      locked = true;
      locked_kind = used_kind;
      locked_size = used_size;
      locked_circles_flags = used_circles_flags;
      return true;
    }
  }

  used_kind = PatternKind::Chessboard;
  used_size = configured_size;
  used_circles_flags = 0;
  return false;
}

std::vector<cv::Point3f> centers_3d(const cv::Size & pattern_size, const float center_distance)
{
  std::vector<cv::Point3f> centers_3d;

  for (int i = 0; i < pattern_size.height; i++)
    for (int j = 0; j < pattern_size.width; j++)
      centers_3d.push_back({j * center_distance, i * center_distance, 0});

  return centers_3d;
}

Eigen::Quaterniond read_q(const std::string & q_path)
{
  std::ifstream q_file(q_path);
  double w, x, y, z;
  q_file >> w >> x >> y >> z;
  return {w, x, y, z};
}

void load(
  const std::string & input_folder, const std::string & config_path,
  std::vector<double> & R_gimbal2imubody_data, std::vector<cv::Mat> & R_gimbal2world_list,
  std::vector<cv::Mat> & t_gimbal2world_list, std::vector<cv::Mat> & rvecs,
  std::vector<cv::Mat> & tvecs)
{
  // 读取yaml参数
  auto yaml = YAML::LoadFile(config_path);
  auto pattern_cols = yaml["pattern_cols"].as<int>();
  auto pattern_rows = yaml["pattern_rows"].as<int>();
  auto center_distance_mm = yaml["center_distance_mm"].as<double>();
  R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();

  cv::Size pattern_size(pattern_cols, pattern_rows);
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_gimbal2imubody(R_gimbal2imubody_data.data());
  cv::Matx33d camera_matrix(camera_matrix_data.data());
  cv::Mat distort_coeffs(distort_coeffs_data);

  bool locked = false;
  PatternKind locked_kind = PatternKind::Chessboard;
  cv::Size locked_size;
  int locked_circles_flags = 0;

  for (int i = 1; true; i++) {
    // 读取图片和对应四元数
    auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
    auto q_path = fmt::format("{}/{}.txt", input_folder, i);

    // 检查文件是否存在且是常规文件
    if (!std::filesystem::exists(img_path) || !std::filesystem::is_regular_file(img_path)) break;
    if (!std::filesystem::exists(q_path) || !std::filesystem::is_regular_file(q_path)) break;

    auto img = cv::imread(img_path);
    Eigen::Quaterniond q = read_q(q_path);
    if (img.empty()) break;

    // 计算云台的欧拉角
    Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
    Eigen::Matrix3d R_gimbal2world =
      R_gimbal2imubody.transpose() * R_imubody2imuabs * R_gimbal2imubody;
    Eigen::Vector3d ypr = tools::eulers(R_gimbal2world, 2, 1, 0) * 57.3;  // degree

    // 在图片上显示云台的欧拉角，用来检验R_gimbal2imubody是否正确
    auto drawing = img.clone();
    tools::draw_text(drawing, fmt::format("yaw   {:.2f}", ypr[0]), {40, 40}, {0, 0, 255});
    tools::draw_text(drawing, fmt::format("pitch {:.2f}", ypr[1]), {40, 80}, {0, 0, 255});
    tools::draw_text(drawing, fmt::format("roll  {:.2f}", ypr[2]), {40, 120}, {0, 0, 255});

    // 识别标定板
    std::vector<cv::Point2f> centers_2d;
    PatternKind used_kind = PatternKind::Chessboard;
    cv::Size used_size = pattern_size;
    int used_circles_flags = 0;
    auto success = detect_pattern_locked(
      img, pattern_size, locked, locked_kind, locked_size, locked_circles_flags, centers_2d,
      used_kind, used_size, used_circles_flags);

    // 显示识别结果
    cv::drawChessboardCorners(drawing, used_size, centers_2d, success);
    cv::resize(drawing, drawing, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("Press any to continue", drawing);
    cv::waitKey(0);

    // 输出识别结果
    if (success) {
      if (used_kind == PatternKind::Chessboard) {
        fmt::print(
          "[success] {} (chessboard {}x{})\n", img_path, used_size.width, used_size.height);
      } else {
        fmt::print(
          "[success] {} (circles {}x{}, flags={})\n", img_path, used_size.width,
          used_size.height, used_circles_flags);
      }
    } else {
      fmt::print("[failure] {}\n", img_path);
    }
    if (!success) continue;

    // 计算所需的数据
    cv::Mat t_gimbal2world = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    cv::Mat R_gimbal2world_cv;
    cv::eigen2cv(R_gimbal2world, R_gimbal2world_cv);
    cv::Mat rvec, tvec;
    auto centers_3d_ = centers_3d(used_size, center_distance_mm);
    cv::solvePnP(
      centers_3d_, centers_2d, camera_matrix, distort_coeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE);

    // 记录所需的数据
    R_gimbal2world_list.emplace_back(R_gimbal2world_cv);
    t_gimbal2world_list.emplace_back(t_gimbal2world);
    rvecs.emplace_back(rvec);
    tvecs.emplace_back(tvec);
  }
}

void print_yaml(
  const std::vector<double> & R_gimbal2imubody_data, const cv::Mat & R_camera2gimbal,
  const cv::Mat & t_camera2gimbal, const Eigen::Vector3d & ypr)
{
  YAML::Emitter result;
  std::vector<double> R_camera2gimbal_data(
    R_camera2gimbal.begin<double>(), R_camera2gimbal.end<double>());
  std::vector<double> t_camera2gimbal_data(
    t_camera2gimbal.begin<double>(), t_camera2gimbal.end<double>());

  result << YAML::BeginMap;
  result << YAML::Key << "R_gimbal2imubody";
  result << YAML::Value << YAML::Flow << R_gimbal2imubody_data;
  result << YAML::Newline;
  result << YAML::Newline;
  result << YAML::Comment(fmt::format(
    "相机同理想情况的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree", ypr[0], ypr[1], ypr[2]));
  result << YAML::Key << "R_camera2gimbal";
  result << YAML::Value << YAML::Flow << R_camera2gimbal_data;
  result << YAML::Key << "t_camera2gimbal";
  result << YAML::Value << YAML::Flow << t_camera2gimbal_data;
  result << YAML::Newline;
  result << YAML::EndMap;

  fmt::print("\n{}\n", result.c_str());
}

int main(int argc, char * argv[])
{
  auto args = parse_args(argc, argv);
  auto & input_folder = args.input_folder;
  auto & config_path = args.config_path;

  if (input_folder.empty()) {
    fmt::print(stderr, "Error: input folder not specified. Use -i <folder> or a positional <folder>\n");
    print_usage(argv[0]);
    return 1;
  }

  if (std::filesystem::is_directory(config_path)) {
    fmt::print(stderr, "Error: config-path points to a directory: {}\n", config_path);
    return 1;
  }
  if (!std::filesystem::exists(config_path)) {
    fmt::print(stderr, "Error: config file not found: {}\n", config_path);
    return 1;
  }

  // 从输入文件夹中加载标定所需的数据
  std::vector<double> R_gimbal2imubody_data;
  std::vector<cv::Mat> R_gimbal2world_list, t_gimbal2world_list;
  std::vector<cv::Mat> rvecs, tvecs;
  load(
    input_folder, config_path, R_gimbal2imubody_data, R_gimbal2world_list, t_gimbal2world_list,
    rvecs, tvecs);

  if (R_gimbal2world_list.empty() || rvecs.empty()) {
    fmt::print(stderr, "Error: no valid samples loaded from: {}\n", input_folder);
    fmt::print(stderr, "Check that it contains 1.jpg/1.txt, 2.jpg/2.txt, ... and that the pattern can be detected.\n");
    return 1;
  }

  // 手眼标定
  cv::Mat R_camera2gimbal, t_camera2gimbal;
  cv::calibrateHandEye(
    R_gimbal2world_list, t_gimbal2world_list, rvecs, tvecs, R_camera2gimbal, t_camera2gimbal);
  t_camera2gimbal /= 1e3;  // mm to m

  // 计算相机同理想情况的偏角
  Eigen::Matrix3d R_camera2gimbal_eigen;
  cv::cv2eigen(R_camera2gimbal, R_camera2gimbal_eigen);
  Eigen::Matrix3d R_gimbal2ideal{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}};
  Eigen::Matrix3d R_camera2ideal = R_gimbal2ideal * R_camera2gimbal_eigen;
  Eigen::Vector3d ypr = tools::eulers(R_camera2ideal, 1, 0, 2) * 57.3;  // degree

  // 输出yaml
  print_yaml(R_gimbal2imubody_data, R_camera2gimbal, t_camera2gimbal, ypr);
}
