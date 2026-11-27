// Lockstep stdin/stdout relay between a paused gz-sim server and a Python controller.
// Python-binding-free: needs only gz-transport13 / gz-msgs10 (Harmonic).
//
// Commands (one per line on stdin), replies on stdout:
//   D d0 d1 d2 d3 N  publish wheel duties (physical FL FR RL RR), step N physics
//                    iterations, wait for the model pose at the new sim time
//                    -> "P t x y yaw"
//   S x y z yaw      set the model pose (world frame)            -> "OK" | "ERR ..."
//   R                reset the world (paused)                     -> "OK" | "ERR ..."
//   G                latest pose                                  -> "P t x y yaw"
//   Q                quit
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/double_v.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/world_control.pb.h>
#include <gz/transport/Node.hh>

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace {
std::mutex mtx;
std::condition_variable cv;
double poseT = -1, poseX = 0, poseY = 0, poseYaw = 0;
std::string model;

void OnPose(const gz::msgs::Pose_V &msg) {
  for (const auto &p : msg.pose()) {
    if (p.name() != model) continue;
    const auto &q = p.orientation();
    std::lock_guard<std::mutex> lk(mtx);
    // PosePublisher stamps each pose (the Pose_V header is left empty)
    const auto &st = p.has_header() ? p.header().stamp() : msg.header().stamp();
    poseT = st.sec() + st.nsec() * 1e-9;
    poseX = p.position().x();
    poseY = p.position().y();
    poseYaw = std::atan2(2 * (q.w() * q.z() + q.x() * q.y()),
                         1 - 2 * (q.y() * q.y() + q.z() * q.z()));
    cv.notify_all();
    return;
  }
}

void PrintPose() {
  std::lock_guard<std::mutex> lk(mtx);
  std::cout.precision(9);
  std::cout << "P " << poseT << " " << poseX << " " << poseY << " " << poseYaw << std::endl;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: lockstep_bridge <world> <model> <step_s>" << std::endl;
    return 2;
  }
  const std::string world = argv[1];
  model = argv[2];
  if (model == "payload_car" || world == "payload_lab") {
    std::cerr << "payload car has no measured mass or motor model "
                 "(research/tasks/2026-11-27-characterise-payload-car.md); "
                 "regenerate with `python3 generate.py --car freenove ...` "
                 "to drive the archived Freenove car"
              << std::endl;
    return 2;
  }
  const double stepS = std::stod(argv[3]);
  gz::transport::Node node;
  auto duty = node.Advertise<gz::msgs::Double_V>("/model/" + model + "/wheel_duty");
  node.Subscribe("/model/" + model + "/pose", OnPose);
  const std::string control = "/world/" + world + "/control";
  const std::string setPose = "/world/" + world + "/set_pose";

  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    char c;
    if (!(in >> c)) continue;
    if (c == 'Q') break;
    if (c == 'G') { PrintPose(); continue; }
    if (c == 'D') {
      gz::msgs::Double_V d;
      for (int i = 0; i < 4; ++i) { double v; in >> v; d.add_data(v); }
      int n; in >> n;
      if (!duty.HasConnections()) {
        for (int i = 0; i < 200 && !duty.HasConnections(); ++i)
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      double t0;
      { std::lock_guard<std::mutex> lk(mtx); t0 = poseT; }
      duty.Publish(d);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      gz::msgs::WorldControl req;
      req.set_pause(true);
      req.set_multi_step(n);
      gz::msgs::Boolean rep;
      bool ok = false;
      bool sent = node.Request(control, req, 5000, rep, ok);
      if (!sent || !ok) { std::cout << "ERR rpc" << std::endl; continue; }
      if (!rep.data()) { std::cout << "ERR rejected" << std::endl; continue; }
      std::unique_lock<std::mutex> lk(mtx);
      const double target = t0 + n * stepS - 0.5 * stepS;
      bool got = cv.wait_for(lk, std::chrono::seconds(10), [&] { return poseT >= target; });
      lk.unlock();
      if (!got) { std::cout << "ERR pose_timeout" << std::endl; continue; }
      PrintPose();
      continue;
    }
    if (c == 'S') {
      double x, y, z, yaw; in >> x >> y >> z >> yaw;
      gz::msgs::Double_V zeros;
      for (int i = 0; i < 4; ++i) zeros.add_data(0.0);
      if (duty.HasConnections()) {
        duty.Publish(zeros);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      gz::msgs::Pose req;
      req.set_name(model);
      req.mutable_position()->set_x(x);
      req.mutable_position()->set_y(y);
      req.mutable_position()->set_z(z);
      req.mutable_orientation()->set_w(std::cos(yaw / 2));
      req.mutable_orientation()->set_z(std::sin(yaw / 2));
      gz::msgs::Boolean rep;
      bool ok = false;
      bool sent = node.Request(setPose, req, 5000, rep, ok);
      if (!sent || !ok) { std::cout << "ERR rpc" << std::endl; continue; }
      if (!rep.data()) { std::cout << "ERR rejected" << std::endl; continue; }
      std::cout << "OK" << std::endl;
      continue;
    }
    if (c == 'R') {
      gz::msgs::WorldControl req;
      req.set_pause(true);
      req.mutable_reset()->set_all(true);
      gz::msgs::Boolean rep;
      bool ok = false;
      bool sent = node.Request(control, req, 5000, rep, ok);
      if (!sent || !ok) { std::cout << "ERR rpc" << std::endl; continue; }
      if (!rep.data()) { std::cout << "ERR rejected" << std::endl; continue; }
      { std::lock_guard<std::mutex> lk(mtx); poseT = 0; }  // reset rewinds the sim clock
      std::cout << "OK" << std::endl;
      continue;
    }
    std::cout << "ERR unknown" << std::endl;
  }
  return 0;
}
