#ifndef IO__PBLISH2NAV_HPP
#define IO__PBLISH2NAV_HPP

#include <Eigen/Dense>  // For Eigen::Vector3d
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "communicate_2025/msg/serial_info.hpp"

namespace io
{
class Publish2Nav : public rclcpp::Node
{
public:
  Publish2Nav();

  ~Publish2Nav();

  void start();

  void send_data(const Eigen::Vector4d & data);

private:
  // ROS2 发布者
  rclcpp::Publisher<communicate_2025::msg::SerialInfo>::SharedPtr publisher_;
};

}  // namespace io

#endif  // Publish2Nav_HPP_
