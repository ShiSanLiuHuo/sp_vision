#include "publish2nav.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <memory>
#include <thread>

#include "tools/logger.hpp"

namespace io
{

Publish2Nav::Publish2Nav() : Node("auto_aim_target_pos_publisher")
{
  publisher_ = this->create_publisher<communicate_2025::msg::SerialInfo>("/shoot_info", 10);

  RCLCPP_INFO(this->get_logger(), "auto_aim_target_pos_publisher node initialized.");
}

Publish2Nav::~Publish2Nav()
{
  RCLCPP_INFO(this->get_logger(), "auto_aim_target_pos_publisher node shutting down.");
}

void Publish2Nav::send_data(const Eigen::Vector4d & target_pos)
{
  // 创建消息
  auto message = std::make_shared<communicate_2025::msg::SerialInfo>();

  // 将 Eigen::Vector3d 数据转换为字符串并存储在消息中
  // 将 Eigen::Vector4d 数据转换为消息字段
  message->yaw = static_cast<float>(target_pos[0]);
  message->pitch = static_cast<float>(target_pos[1]);
  
  // is_find 是 std_msgs/Char 类型，需要将double转换为char
  // 通常使用 0/1 来表示 false/true
  message->is_find.data = static_cast<uint8_t>(target_pos[3] != 0.0 ? 1 : 0);
 
  // 发布消息
  publisher_->publish(*message);
 


  // 发布消息
  publisher_->publish(*message);

  // RCLCPP_INFO(
  //   this->get_logger(), "auto_aim_target_pos_publisher node sent message: '%s'",
  //   message->data.c_str());
}

void Publish2Nav::start()
{
  RCLCPP_INFO(this->get_logger(), "auto_aim_target_pos_publisher node starting to spin...");
  rclcpp::spin(this->shared_from_this());
}

}  // namespace io
