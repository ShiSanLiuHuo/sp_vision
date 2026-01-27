#ifndef IO__COMMAND_HPP
#define IO__COMMAND_HPP

namespace io
{
struct Command
{
  bool control;
  bool shoot;
  double yaw;
  double pitch;
  double yaw_vel;
  double pitch_vel;
  double horizon_distance = 0;  //无人机专有
  double distance=0;
};

}  // namespace io

#endif  // IO__COMMAND_HPP