#include <cstddef>
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>

#include <fmt/core.h>
#include <fmt/format.h> // Explicit formatting support

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <optional>
#include "rclcpp/rclcpp.hpp"
#include "io/ros2/subscribe2nav.hpp"
#include "io/ros2/publish2nav.hpp"

#include "io/camera.hpp"
#include "io/ros2/ros2.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

using namespace std::chrono;

const std::string keys =
    "{help h usage ? |      | 输出命令行参数说明}"
    "{@config-path   | configs/standard3.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char* argv[]) {
    // 手工解析参数：支持位置参数 config-path，和可选 -v/--video <path>
    std::string config_path = "configs/standard3.yaml";
    std::string video_path;
    bool use_video = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help" || a == "-?") {
            std::cout << "Usage: " << argv[0] << " [config-path] [-v video_path]\n";
            return 0;
        }
        if (a == "-v" || a == "--video") {
            if (i + 1 < argc) { video_path = argv[++i]; use_video = true; continue; }
            std::cerr << "-v requires an argument\n";
            return 1;
        }
        if (a.size() > 0 && a[0] != '-') {
            if (config_path == "configs/standard3.yaml") config_path = a;
        }
    }

    double jump_pitch_up_duration = 0.0;
    double jump_pitch_down_duration = 0.0;
    double decision_speed = 0.0;
    bool use_identity_pose = false;
    double target_jump_angle_threshold_deg = 5.0;   // 判定为"大跳变"的阈值（度）
    double target_stabilize_alpha = 0.3;             // 平滑系数，越小越平滑（推荐0.1~0.5）
    bool enable_target_stabilize = true;             // 是否启用平滑过渡
    try {
        auto yaml = YAML::LoadFile(config_path);
        if (yaml["jump_pitch_up_duration"].IsDefined()) {
            jump_pitch_up_duration = yaml["jump_pitch_up_duration"].as<double>();
        }
        if (yaml["jump_pitch_down_duration"].IsDefined()) {
            jump_pitch_down_duration = yaml["jump_pitch_down_duration"].as<double>();
        }
        if (yaml["decision_speed"].IsDefined()) {
            decision_speed = yaml["decision_speed"].as<double>();
        }
        if (yaml["use_identity_pose"].IsDefined()) {
            use_identity_pose = yaml["use_identity_pose"].as<bool>();
        }
        if (yaml["target_jump_angle_threshold_deg"].IsDefined()) {
            target_jump_angle_threshold_deg = yaml["target_jump_angle_threshold_deg"].as<double>();
        }
        if (yaml["target_stabilize_alpha"].IsDefined()) {
            target_stabilize_alpha = yaml["target_stabilize_alpha"].as<double>();
        }
        if (yaml["enable_target_stabilize"].IsDefined()) {
            enable_target_stabilize = yaml["enable_target_stabilize"].as<bool>();
        }
    } catch (const YAML::Exception & e) {
        tools::logger()->warn("Failed to read configuration: {}", e.what());
    }

    if (use_identity_pose) {
        tools::logger()->warn("[Debug] use_identity_pose=true, IMU quaternion from CBoard is ignored.");
    }

    tools::Exiter exiter;
    tools::Plotter plotter;
    tools::Recorder recorder;

    // Replace direct CBoard usage with ROS2-based interface
    // Read minimal cboard-related config fields from YAML so behavior remains similar
    double cboard_bullet_speed = 21.0;
    bool cboard_use_default_bullet_speed = false;
    bool phoenix_angles_in_degrees = false;
    double imu_yaw_offset_rad = 0.0;
    double imu_pitch_offset_rad = 0.0;
    try {
        auto yaml = YAML::LoadFile(config_path);
        if (yaml["bullet_speed"]) cboard_bullet_speed = yaml["bullet_speed"].as<double>();
        if (yaml["use_default_bullet_speed"]) cboard_use_default_bullet_speed = yaml["use_default_bullet_speed"].as<bool>();
        if (yaml["phoenix_angle_unit"]) {
            auto unit = yaml["phoenix_angle_unit"].as<std::string>();
            for (auto &c: unit) c = static_cast<char>(std::tolower(c));
            phoenix_angles_in_degrees = (unit == "deg" || unit == "degree" || unit == "degrees");
        }
        if (yaml["imu_yaw_offset_deg"]) imu_yaw_offset_rad = yaml["imu_yaw_offset_deg"].as<double>() * M_PI / 180.0;
        if (yaml["imu_pitch_offset_deg"]) imu_pitch_offset_rad = yaml["imu_pitch_offset_deg"].as<double>() * M_PI / 180.0;
        if (yaml["imu_yaw_offset_rad"]) imu_yaw_offset_rad = yaml["imu_yaw_offset_rad"].as<double>();
        if (yaml["imu_pitch_offset_rad"]) imu_pitch_offset_rad = yaml["imu_pitch_offset_rad"].as<double>();
    } catch (const YAML::Exception & e) {
        tools::logger()->warn("Failed to read cboard config from {}: {}", config_path, e.what());
    }

    // Only construct a camera if we are not in video playback mode
    std::unique_ptr<io::Camera> camera;
    std::unique_ptr<cv::VideoCapture> video_cap;
    if (!use_video) {
        camera = std::make_unique<io::Camera>(config_path);
    } else {
        video_cap = std::make_unique<cv::VideoCapture>(video_path);
        if (!video_cap->isOpened()) {
            std::cerr << "Failed to open video: " << video_path << std::endl;
            return 1;
        }
        tools::logger()->info("Using video input: {}", video_path);
    }

    // Initialize ROS2 and IO wrappers (non-blocking): Subscribe to autoaim and publish commands
    rclcpp::init(argc, argv);
    auto nav_sub = std::make_shared<io::Subscribe2Nav>();
    auto pub_node = std::make_shared<io::Publish2Nav>();
    std::thread nav_thread([nav_sub]() { nav_sub->start(); });
    nav_thread.detach();
    std::thread pub_thread([pub_node]() { pub_node->start(); });
    pub_thread.detach();

    // Enable YOLO debug display to help diagnose detection issues (shows detection window)
    auto_aim::YOLO detector(config_path, true);

    // Log YOLO/model configuration for debugging
    try {
        auto yaml = YAML::LoadFile(config_path);
        if (yaml["yolo_name"]) {
            auto yolo_name = yaml["yolo_name"].as<std::string>();
            tools::logger()->info("yolo_name: {}", yolo_name);
            std::string model_key;
            if (yolo_name == "yolov8") model_key = "yolov8_model_path";
            else if (yolo_name == "yolov5") model_key = "yolov5_model_path";
            else if (yolo_name == "yolo11") model_key = "yolo11_model_path";
            if (!model_key.empty() && yaml[model_key]) {
                tools::logger()->info("{}: {}", model_key, yaml[model_key].as<std::string>());
            }
        }
        if (yaml["min_confidence"]) {
            tools::logger()->info("min_confidence: {}", yaml["min_confidence"].as<double>());
        }
        if (yaml["use_roi"]) {
            tools::logger()->info("use_roi: {}", yaml["use_roi"].as<bool>());
            if (yaml["roi"]) {
                try {
                    int rx = yaml["roi"]["x"].as<int>();
                    int ry = yaml["roi"]["y"].as<int>();
                    int rw = yaml["roi"]["width"].as<int>();
                    int rh = yaml["roi"]["height"].as<int>();
                    tools::logger()->info("roi: x={}, y={}, w={}, h={}", rx, ry, rw, rh);
                } catch (...) {
                    ;
                }
            }
        }
    } catch (const YAML::Exception & e) {
        tools::logger()->warn("Failed to read yolo config for logging: {}", e.what());
    }
    auto_aim::Solver solver(config_path);
    auto_aim::Tracker tracker(config_path, solver);
    auto_aim::Planner planner(config_path);
    auto_aim::Aimer aimer(config_path);
    auto_aim::Shooter shooter(config_path);

    cv::Mat img;
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point t;

    // profiling counters (same as standard_mpc_se)
    int debug_log_interval_frames = 200;
    int frames_since_log = 0;
    auto last_log_time = std::chrono::steady_clock::now();
    double acc_cam = 0.0, acc_detect = 0.0, acc_track = 0.0, acc_aim = 0.0, acc_render = 0.0, acc_total = 0.0;

    auto mode       = io::Mode::idle;
    auto last_mode  = io::Mode::idle;
    int frame_count = 0;
    io::Command last_command = {false, false, 0.0, 0.0, 0.0, 0.0};
    int total_armors = 0;  // 总检测到的装甲板数量
    int detected_frames = 0;  // 检测到装甲板的帧数
    


    while (!exiter.exit()) {
        auto frame_start = std::chrono::steady_clock::now();
        auto cam_start = std::chrono::steady_clock::now();
        if (use_video) {
            if (!video_cap->read(img) || img.empty()) {
                // loop back to start
                video_cap->set(cv::CAP_PROP_POS_FRAMES, 0);
                // try to read again; if still fails, sleep and continue
                if (!video_cap->read(img) || img.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
            }
            t = std::chrono::steady_clock::now();
        } else {
            camera->read(img, t);
        }
        double cam_dt = tools::delta_time(std::chrono::steady_clock::now(), cam_start);
    // try to get latest autoaim data from ROS2; if absent, fall back to identity quaternion
        std::optional<io::AutoaimData> maybe = std::nullopt;
        // Subscribe2Nav exposes get_autoaim_data()
        if (nav_sub) {
            maybe = nav_sub->get_autoaim_data();
            tools::logger()->info("maybe has value: {}", maybe.has_value());
        }

        if (use_identity_pose) {
            q = Eigen::Quaterniond::Identity();
        } else if (maybe.has_value()) {
            auto ad = maybe.value();
            double yaw = static_cast<double>(ad.high_gimbal_yaw);
            double pitch = static_cast<double>(ad.pitch);
            if (phoenix_angles_in_degrees) {
                constexpr double kDeg2Rad = M_PI / 180.0;
                yaw *= kDeg2Rad;
                pitch *= kDeg2Rad;
            }
            yaw += imu_yaw_offset_rad;
            pitch += imu_pitch_offset_rad;
            Eigen::AngleAxisd yaw_aa(yaw, Eigen::Vector3d::UnitZ());
            Eigen::AngleAxisd pitch_aa(pitch, Eigen::Vector3d::UnitY());
            q = (yaw_aa * pitch_aa).normalized();
            mode = static_cast<io::Mode>(ad.mode);
        } else {
            q = Eigen::Quaterniond::Identity();
        }

        if (last_mode != mode) {
            tools::logger()->info("Switch to {}", io::MODES[mode].c_str());
            last_mode = mode;
        }

        recorder.record(img, q, t);

        solver.set_R_gimbal2world(q);

    auto yolo_start    = std::chrono::steady_clock::now();
    auto armors        = detector.detect(img);
    double detect_dt = tools::delta_time(std::chrono::steady_clock::now(), yolo_start);
        total_armors += armors.size();  // 累加检测到的装甲板
        if (!armors.empty()) {
            detected_frames++;  // 累加检测成功的帧数
        }
    auto tracker_start = std::chrono::steady_clock::now();
    auto targets       = tracker.track(armors, t);
    double track_dt = tools::delta_time(std::chrono::steady_clock::now(), tracker_start);
    auto aimer_start   = std::chrono::steady_clock::now();
    auto command       = aimer.aim(targets, t, cboard_bullet_speed);
    double aim_dt = tools::delta_time(std::chrono::steady_clock::now(), aimer_start);
        
        if (!targets.empty()) {
            auto plan = planner.plan(targets.front(), cboard_bullet_speed);
            if (plan.control) {
                command.yaw       = plan.yaw;
                command.pitch     = plan.pitch;
                command.yaw_vel   = plan.yaw_vel;
                command.pitch_vel = plan.pitch_vel;
            }
        }

    auto now_tp        = std::chrono::steady_clock::now();

        Eigen::Quaterniond gimbal_q = q;
        Eigen::Vector3d ypr         = tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0);

        command.shoot = shooter.shoot(command, aimer, targets, ypr);

        if (command.control) {
            last_command = command;
        }

        // tools::logger()->info(
        //     "[{}] yolo: {:.1f}ms, tracker: {:.1f}ms, aimer: {:.1f}ms",
        //     frame_count,
        //     tools::delta_time(tracker_start, yolo_start) * 1e3,
        //     tools::delta_time(aimer_start, tracker_start) * 1e3,
        //     tools::delta_time(finish, aimer_start) * 1e3
        // );

        auto yaw                    = ypr[0];

        tools::draw_text(
            img,
            fmt::format(
                "command is {},{:.2f},{:.2f},shoot:{}",
                command.control,
                command.yaw * 57.3,
                command.pitch * 57.3,
                command.shoot
            ),
            { 10, 60 },
            { 154, 50, 205 }
        );
        tools::draw_text(
            img,
            fmt::format("gimbal yaw{:.2f}", yaw * 57.3),
            { 10, 90 },
            { 255, 255, 255 }
        );

        nlohmann::json data;
        data["armor_num"] = armors.size();
        if (!armors.empty()) {
            const auto& armor      = armors.front();
            data["armor_x"]        = armor.xyz_in_world[0];
            data["armor_y"]        = armor.xyz_in_world[1];
            data["armor_yaw"]      = armor.ypr_in_world[0] * 57.3;
            data["armor_yaw_raw"]  = armor.yaw_raw * 57.3;
            data["armor_center_x"] = armor.center_norm.x;
            data["armor_center_y"] = armor.center_norm.y;
        }

        data["gimbal_yaw"] = yaw * 57.3;
        data["cmd_yaw"]    = command.yaw * 57.3;
        data["shoot"]      = command.shoot;

        if (!targets.empty()) {
            auto target                                  = targets.front();
            std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
            std::array<int, 3> outpost_order{0, 1, 2};
            if (target.name == auto_aim::ArmorName::outpost && armor_xyza_list.size() == 3) {
                std::sort(
                    outpost_order.begin(),
                    outpost_order.end(),
                    [&](int a, int b) { return armor_xyza_list[a][2] < armor_xyza_list[b][2]; }
                );
            }

            bool show_jump_up = false;
            bool show_jump_down = false;
            if (target.has_jump_time()) {
                const double abs_w = std::abs(target.ekf_x()[7]);
                if (abs_w >= decision_speed) {
                    auto age = std::chrono::duration<double>(t - target.last_jump_time()).count();
                    auto dir = target.last_jump_dir();
                    if (dir < 0 && jump_pitch_up_duration > 0.0 && age >= 0.0 && age <= jump_pitch_up_duration) {
                        show_jump_up = true;
                    }
                    if (dir > 0 && jump_pitch_down_duration > 0.0 && age >= 0.0 && age <= jump_pitch_down_duration) {
                        show_jump_down = true;
                    }
                }
            }

            for (std::size_t i = 0; i < armor_xyza_list.size(); ++i) {
                const auto& xyza = armor_xyza_list[i];
                auto image_points =
                    solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
                tools::draw_points(img, image_points, { 0, 255, 0 });

                cv::Point2f center{0.0F, 0.0F};
                for (const auto& pt : image_points) {
                    center.x += pt.x;
                    center.y += pt.y;
                }
                center.x /= static_cast<float>(image_points.size());
                center.y /= static_cast<float>(image_points.size());
                tools::draw_text(img, fmt::format("id:{}", i), center, { 255, 255, 0 });

                if (target.name == auto_aim::ArmorName::outpost && armor_xyza_list.size() == 3) {
                    std::string tag = "middle";
                    if (static_cast<int>(i) == outpost_order[0]) tag = "low";
                    if (static_cast<int>(i) == outpost_order[2]) tag = "high";
                    tools::draw_text(img, tag, {center.x, center.y + 18.0F}, { 0, 255, 255 });
                }
            }
            Eigen::VectorXd x = target.ekf_x();
            std::vector<cv::Point3f> center_pt = {{static_cast<float>(x[0]), static_cast<float>(x[2]), static_cast<float>(x[4])}};
            auto center_img_pts = solver.world2pixel(center_pt);
            if (!center_img_pts.empty()) {
                cv::circle(img, center_img_pts[0], 5, {255, 255, 0}, -1); // Cyan circle
            }
            auto aim_point           = aimer.debug_aim_point;
            Eigen::Vector4d aim_xyza = aim_point.xyza;
            auto image_points =
                solver
                    .reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
            if (aim_point.valid) {
                tools::draw_points(img, image_points, { 0, 0, 255 });

                cv::Point2f aim_center{0.0F, 0.0F};
                for (const auto& pt : image_points) {
                    aim_center.x += pt.x;
                    aim_center.y += pt.y;
                }
                aim_center.x /= static_cast<float>(image_points.size());
                aim_center.y /= static_cast<float>(image_points.size());
                if (show_jump_up) {
                    tools::draw_text(img, "Up", {aim_center.x, aim_center.y - 14.0F}, { 0, 165, 255 });
                }
                if (show_jump_down) {
                    tools::draw_text(img, "Down", {aim_center.x, aim_center.y - 14.0F}, { 0, 0, 255 });
                }
            }

            data["x"]         = x[0];
            data["vx"]        = x[1];
            data["y"]         = x[2];
            data["vy"]        = x[3];
            data["z"]         = x[4];
            data["vz"]        = x[5];
            data["a"]         = x[6] * 57.3;
            data["w"]         = x[7];
            data["r"]         = x[8];
            data["l"]         = x[9];
            data["h"]         = x[10];
            data["last_id"]   = target.last_id;

            auto ekf = target.ekf();

            data["residual_yaw"]        = ekf.data.at("residual_yaw");
            data["residual_pitch"]      = ekf.data.at("residual_pitch");
            data["residual_distance"]   = ekf.data.at("residual_distance");
            data["residual_angle"]      = ekf.data.at("residual_angle");
            data["nis"]                 = ekf.data.at("nis");
            data["nees"]                = ekf.data.at("nees");
            data["nis_fail"]            = ekf.data.at("nis_fail");
            data["nees_fail"]           = ekf.data.at("nees_fail");
            data["recent_nis_failures"] = ekf.data.at("recent_nis_failures");
        }

        // compute totals and render estimate
        double total_dt = tools::delta_time(now_tp, frame_start);
        double render_dt = total_dt - (cam_dt + detect_dt + track_dt + aim_dt);
        if (render_dt < 0.0) render_dt = 0.0;

        // accumulate and overlay
        acc_cam += cam_dt; acc_detect += detect_dt; acc_track += track_dt; acc_aim += aim_dt; acc_render += render_dt; acc_total += total_dt;
        frames_since_log++;
        double inst_fps = total_dt > 1e-6 ? 1.0 / total_dt : 0.0;
        tools::draw_text(img, fmt::format("FPS: {:.1f}", inst_fps), {10, 30}, {0, 255, 0});

        // periodic logging
        auto now = std::chrono::steady_clock::now();
        double elapsed_since_log = std::chrono::duration<double>(now - last_log_time).count();
        if (frames_since_log >= debug_log_interval_frames || elapsed_since_log >= 5.0) {
            double avg_cam = acc_cam / frames_since_log * 1e3;
            double avg_detect = acc_detect / frames_since_log * 1e3;
            double avg_track = acc_track / frames_since_log * 1e3;
            double avg_aim = acc_aim / frames_since_log * 1e3;
            double avg_render = acc_render / frames_since_log * 1e3;
            double avg_total = acc_total / frames_since_log * 1e3;
            double fps = frames_since_log / (elapsed_since_log > 0 ? elapsed_since_log : 1.0);
            tools::logger()->info("PROFILE avg over {} frames: fps={:.1f}, total={:.2f}ms, cam={:.2f}ms, detect={:.2f}ms, track={:.2f}ms, aim={:.2f}ms, render={:.2f}ms",
                                 frames_since_log, fps, avg_total, avg_cam, avg_detect, avg_track, avg_aim, avg_render);
            frames_since_log = 0; acc_cam = acc_detect = acc_track = acc_aim = acc_render = acc_total = 0.0; last_log_time = now;
        }

        plotter.plot(data);

        cv::resize(img, img, {}, 0.9, 0.9);
        cv::imshow("reprojection", img);
        auto key = cv::waitKey(1);
        if (key == 'q')
            break;

        // 平滑过渡逻辑：当目标位置跳变过大时，逐帧过渡而不是直接跳变
        if (enable_target_stabilize && command.control) {
            double delta_yaw = std::abs(command.yaw - last_command.yaw) * 57.3;   // 转为度
            double delta_pitch = std::abs(command.pitch - last_command.pitch) * 57.3;
            
            // 如果yaw或pitch跳变超过阈值，应用指数平滑
            if (delta_yaw > target_jump_angle_threshold_deg || delta_pitch > target_jump_angle_threshold_deg) {
                command.yaw = (1.0 - target_stabilize_alpha) * last_command.yaw + target_stabilize_alpha * command.yaw;
                command.pitch = (1.0 - target_stabilize_alpha) * last_command.pitch + target_stabilize_alpha * command.pitch;
            }
        }

        // send command via ROS2 publisher node
        if (pub_node) {
            Eigen::Vector4d out;
            out[0] = command.yaw;
            out[1] = command.pitch;
            out[2] = 0.0; // reserved
            out[3] = command.control ? 1.0 : 0.0; // is_find flag
            pub_node->send_data(out);
        }
        frame_count++;
    }
        // 在程序结束时输出识别率
    if (frame_count > 0) {
        double avg_armors_per_frame = static_cast<double>(total_armors) / frame_count;
        double detection_rate = 100.0 * detected_frames / frame_count;
        tools::logger()->info(
            "Recognition Summary: Total frames: {}, Total armors detected: {}, "
            "Average armors per frame: {:.2f}, Detection rate: {:.2f}% ({}/{} frames)",
            frame_count, total_armors, avg_armors_per_frame, detection_rate, detected_frames, frame_count
        );
    }

    return 0;
}