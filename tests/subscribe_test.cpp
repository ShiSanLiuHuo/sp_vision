#include <rclcpp/rclcpp.hpp>
#include <thread>

#include "io/ros2/ros2.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"

int main(int argc, char ** argv)
{
  tools::Exiter exiter;
  io::ROS2 ros2;

  int i = 0;
  while (!exiter.exit()) {
    auto x = ros2.get_autoaim_data();
    // tools::logger()->info("invincible enemy ids size is{}", x.size());
    if (x.has_value()) {
      tools::logger()->info("yaw: {}, pitch: {}, enemy_team_color: {}, mode: {}, rune_flag: {}, low_gimbal_yaw: {}", 
        x.value().high_gimbal_yaw, x.value().pitch, x.value().enemy_team_color, x.value().mode, x.value().rune_flag, x.value().low_gimbal_yaw);
    }
    // i++;

    std::this_thread::sleep_for(std::chrono::microseconds(500));
    // if (i > 1000) break;
  }
  return 0;
}
