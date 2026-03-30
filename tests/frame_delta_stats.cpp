#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

namespace
{
struct Stats
{
  double mean = 0.0;
  double variance_population = 0.0;
  double variance_sample = 0.0;
  double min = 0.0;
  double max = 0.0;
  double range = 0.0;
};

Stats calc_stats(const std::vector<double> & values)
{
  Stats s;
  if (values.empty()) {
    return s;
  }

  double sum = 0.0;
  s.min = values.front();
  s.max = values.front();
  for (double v : values) sum += v;
  s.mean = sum / static_cast<double>(values.size());

  double sq_sum = 0.0;
  for (double v : values) {
    if (v < s.min) s.min = v;
    if (v > s.max) s.max = v;
    const double d = v - s.mean;
    sq_sum += d * d;
  }

  s.variance_population = sq_sum / static_cast<double>(values.size());
  if (values.size() > 1) {
    s.variance_sample = sq_sum / static_cast<double>(values.size() - 1);
  }
  s.range = s.max - s.min;
  return s;
}

bool parse_delta(const std::string & line, double & delta)
{
  // 支持格式："... delta=-2 ..."
  static const std::regex with_key(R"(delta=([-+]?\d+))");
  std::smatch m;
  if (std::regex_search(line, m, with_key)) {
    delta = std::stod(m[1].str());
    return true;
  }

  // 兼容：整行只有一个数字
  static const std::regex number_only(R"(^\s*([-+]?\d+(?:\.\d+)?)\s*$)");
  if (std::regex_match(line, m, number_only)) {
    delta = std::stod(m[1].str());
    return true;
  }

  return false;
}
}  // namespace

int main(int argc, char ** argv)
{
  std::string file_path = "records/frame_count_delta.txt";
  if (argc > 1) {
    file_path = argv[1];
  }

  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open file: " << file_path << std::endl;
    return 1;
  }

  std::vector<double> deltas;
  std::string line;
  while (std::getline(ifs, line)) {
    double delta = 0.0;
    if (parse_delta(line, delta)) {
      deltas.push_back(delta);
    }
  }

  if (deltas.empty()) {
    std::cerr << "No valid delta found in file: " << file_path << std::endl;
    return 2;
  }

  const auto stats = calc_stats(deltas);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "file: " << file_path << std::endl;
  std::cout << "count: " << deltas.size() << std::endl;
  std::cout << "mean: " << stats.mean << std::endl;
  std::cout << "min: " << stats.min << std::endl;
  std::cout << "max: " << stats.max << std::endl;
  std::cout << "range: " << stats.range << std::endl;
  std::cout << "variance(population): " << stats.variance_population << std::endl;
  std::cout << "variance(sample): " << stats.variance_sample << std::endl;

  return 0;
}
