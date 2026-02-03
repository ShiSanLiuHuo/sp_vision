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

#include "io/camera.hpp"
#include "io/cboard.hpp"
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
    cv::CommandLineParser cli(argc, argv, keys);
    auto config_path = cli.get<std::string>(0);
    if (cli.has("help") || config_path.empty()) {
        cli.printMessage();
        return 0;
    }

    tools::Exiter exiter;
    tools::Plotter plotter;
    tools::Recorder recorder;

    io::CBoard cboard(config_path);
    io::Camera camera(config_path);

    auto_aim::YOLO detector(config_path, false);
    auto_aim::Solver solver(config_path);
    auto_aim::Tracker tracker(config_path, solver);
    auto_aim::Planner planner(config_path);
    auto_aim::Aimer aimer(config_path);
    auto_aim::Shooter shooter(config_path);

    cv::Mat img;
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point t;

    auto mode       = io::Mode::idle;
    auto last_mode  = io::Mode::idle;
    int frame_count = 0;
    io::Command last_command = {false, false, 0.0, 0.0, 0.0, 0.0};
    int total_armors = 0;  // 总检测到的装甲板数量
    int detected_frames = 0;  // 检测到装甲板的帧数
    


    while (!exiter.exit()) {
        camera.read(img, t);
        q    = cboard.imu_at(t - 1ms);
        mode = cboard.mode;

        if (last_mode != mode) {
            tools::logger()->info("Switch to {}", io::MODES[mode].c_str());
            last_mode = mode;
        }

        recorder.record(img, q, t);

        solver.set_R_gimbal2world(q);

        auto yolo_start    = std::chrono::steady_clock::now();
        auto armors        = detector.detect(img);
        total_armors += armors.size();  // 累加检测到的装甲板
        if (!armors.empty()) {
            detected_frames++;  // 累加检测成功的帧数
        }
        auto tracker_start = std::chrono::steady_clock::now();
        auto targets       = tracker.track(armors, t);
        auto aimer_start   = std::chrono::steady_clock::now();
        auto command       = aimer.aim(targets, t, cboard.bullet_speed);
        
        if (!targets.empty()) {
            auto plan = planner.plan(targets.front(), cboard.bullet_speed);
            if (plan.control) {
                command.yaw       = plan.yaw;
                command.pitch     = plan.pitch;
                command.yaw_vel   = plan.yaw_vel;
                command.pitch_vel = plan.pitch_vel;
            }
        }

        auto finish        = std::chrono::steady_clock::now();

        Eigen::Quaterniond gimbal_q = q;
        Eigen::Vector3d ypr         = tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0);

        command.shoot = shooter.shoot(command, aimer, targets, ypr);

        if (command.control) {
            last_command = command;
        }

        tools::logger()->info(
            "[{}] yolo: {:.1f}ms, tracker: {:.1f}ms, aimer: {:.1f}ms",
            frame_count,
            tools::delta_time(tracker_start, yolo_start) * 1e3,
            tools::delta_time(aimer_start, tracker_start) * 1e3,
            tools::delta_time(finish, aimer_start) * 1e3
        );

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
                    std::string tag = "中";
                    if (static_cast<int>(i) == outpost_order[0]) tag = "下";
                    if (static_cast<int>(i) == outpost_order[2]) tag = "上";
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

        plotter.plot(data);

        cv::resize(img, img, {}, 0.8, 0.8);
        // cv::imshow("reprojection", img);
        auto key = cv::waitKey(1);
        if (key == 'q')
            break;

        cboard.send(command);
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