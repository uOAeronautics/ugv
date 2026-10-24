// Reference TT gearmotor dynamics at the gearbox output; no hardware access.
#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <vector>
#include <sstream>
#include <random>
#include <gz/msgs/twist.pb.h>
#include <gz/msgs/double_v.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/components/JointVelocity.hh>
#include <gz/sim/components/Inertial.hh>
#include <gz/sim/components/JointForceCmd.hh>
#include <gz/transport/Node.hh>

namespace ugv
{
class TTMotorSystem final : public gz::sim::System,
    public gz::sim::ISystemConfigure, public gz::sim::ISystemPreUpdate,
    public gz::sim::ISystemReset
{
  gz::sim::Model model{gz::sim::kNullEntity};
  gz::transport::Node node;
  gz::transport::Node::Publisher telemetry;
  std::array<gz::sim::Entity,4> joints{};
  const std::array<std::string,4> names{{"front_left_wheel_joint",
      "front_right_wheel_joint", "rear_left_wheel_joint", "rear_right_wheel_joint"}};
  std::mutex mutex;
  double lateral{0}, wheelbase{.1397};
  std::array<double,4> wheelDuty{};
  bool mecanum{false}, fourDuty{false};
  std::array<double,4> wheelGain{{1,1,1,1}}, wheelReverseGain{{1,1,1,1}};
  std::array<double,4> wheelExponent{{1,1,1,1}};
  std::array<double,4> wheelDeadzone{}, wheelDriveTau{}, wheelBrakeTau{}, gainStddev{};
  std::array<double,4> sampledGain{{1,1,1,1}};
  std::mt19937 random{0};

  void SampleParameters()
  {
    std::normal_distribution<double> normal(0.,1.);
    for(size_t i=0;i<4;++i)
      sampledGain[i]=std::max(.05,1.+gainStddev[i]*normal(random));
  }

  void ReadFour(const std::shared_ptr<const sdf::Element> &sdf,
                const std::string &key, std::array<double,4> &values,
                double maximum)
  {
    if(!sdf->HasElement(key)) return;
    std::istringstream stream(sdf->Get<std::string>(key));
    for(auto &value:values)
      if(!(stream>>value) || !std::isfinite(value) || value<0 || value>maximum)
        throw std::runtime_error(key+" needs four bounded nonnegative values: FL FR RL RR");
    double extra;
    if(stream>>extra) throw std::runtime_error(key+" needs exactly four values");
  }

  double linear{0}, angular{0}, radius{.03175}, track{.12065};
  double stall{.0784532}, omega0{26.17994}, feedback{.06};
  double integralGain{.6}, brakeDamping{.06}, brakeResistance{.01}, steeringBrakeTau{0};
  std::array<double,4> brakeEngagement{};
  std::array<double,4> wheelInertia{{1e-5,1e-5,1e-5,1e-5}};
  double pwmLeft{0}, pwmRight{0}, leftVoltageGain{1.475}, rightVoltageGain{1.475};
  bool pwmMode{false};
  std::chrono::steady_clock::time_point lastCommand{};
  std::array<double,4> integral{};
  std::array<std::vector<std::pair<double,double>>,9> pwmMap;
  std::array<double,9> riseTau{};
  std::array<double,4> filteredDuty{};
  std::chrono::steady_clock::duration lastPub{};

  public: void Configure(const gz::sim::Entity &entity,
      const std::shared_ptr<const sdf::Element> &sdf,
      gz::sim::EntityComponentManager &, gz::sim::EventManager &) override
  {
    model = gz::sim::Model(entity);
    mecanum=sdf->Get<bool>("mecanum",false).first;
    wheelbase=sdf->Get<double>("wheelbase",wheelbase).first;
    ReadFour(sdf,"wheel_voltage_gain",wheelGain,5.);
    ReadFour(sdf,"wheel_reverse_voltage_gain",wheelReverseGain,5.);
    ReadFour(sdf,"wheel_deadzone",wheelDeadzone,.95);
    ReadFour(sdf,"wheel_duty_exponent",wheelExponent,3.);
    for(auto exponent:wheelExponent) if(exponent<=0) throw std::runtime_error("Duty exponent must be positive");
    ReadFour(sdf,"wheel_drive_tau",wheelDriveTau,5.);
    ReadFour(sdf,"wheel_brake_tau",wheelBrakeTau,5.);
    ReadFour(sdf,"wheel_gain_stddev",gainStddev,1.);
    random.seed(sdf->Get<unsigned int>("random_seed",0u).first);
    SampleParameters();
    radius = sdf->Get<double>("wheel_radius", radius).first;
    track = sdf->Get<double>("wheel_separation", track).first;
    stall = sdf->Get<double>("stall_torque", stall).first;
    omega0 = sdf->Get<double>("no_load_speed", omega0).first;
    feedback = sdf->Get<double>("velocity_feedback", feedback).first;
    integralGain = sdf->Get<double>("velocity_integral", integralGain).first;
    brakeDamping=sdf->Get<double>("brake_damping",brakeDamping).first;
    brakeResistance=sdf->Get<double>("brake_resistance",brakeResistance).first;
    steeringBrakeTau=sdf->Get<double>("steering_brake_rise_tau",steeringBrakeTau).first;
    if (steeringBrakeTau<0 || radius <= 0 || track <= 0 || stall <= 0 || omega0 <= 0 || brakeDamping<0 || brakeResistance<0)
      throw std::runtime_error("TT motor parameters must be positive");
    leftVoltageGain=sdf->Get<double>("pwm_left_voltage_gain",leftVoltageGain).first;
    rightVoltageGain=sdf->Get<double>("pwm_right_voltage_gain",rightVoltageGain).first;
    if(sdf->HasElement("empirical_pwm_map"))
    {
      std::istringstream rows(sdf->Get<std::string>("empirical_pwm_map"));
      int f,t;double pwm,effort;
      while(rows>>f>>t>>pwm>>effort)
      {
        if(f < -1 || f > 1 || t < -1 || t > 1 || pwm<0 || pwm>1800 || effort<0 || effort>1)
          throw std::runtime_error("Invalid empirical PWM row");
        pwmMap[(f+1)*3+t+1].push_back({pwm,effort});
      }
      for(auto &curve:pwmMap) std::sort(curve.begin(),curve.end());
    }
    if(sdf->HasElement("pwm_rise_tau"))
    {
      std::istringstream rows(sdf->Get<std::string>("pwm_rise_tau"));
      int f,t;double tau;
      while(rows>>f>>t>>tau)
      {
        if(f < -1 || f > 1 || t < -1 || t > 1 || tau<0 || tau>2)
          throw std::runtime_error("Invalid PWM rise lag");
        riseTau[(f+1)*3+t+1]=tau;
      }
    }
    node.Subscribe("/model/freenove/wheel_duty", &TTMotorSystem::PWM, this);
    node.Subscribe("/model/freenove/cmd_vel", &TTMotorSystem::Command, this);
    telemetry = node.Advertise<gz::msgs::Double_V>("/model/freenove/motor_state");
  }

  double EffectiveDuty(double left,double right)
  {
    auto sign=[](double x){return int(x>1e-9)-int(x < -1e-9);};
    int f=sign(left+right),t=sign(right-left);
    const auto &curve=pwmMap[(f+1)*3+t+1];
    double pwm=std::max(std::abs(left),std::abs(right))*4095.;
    if(curve.empty()) return -1;
    if(pwm<=curve.front().first) return curve.front().second;
    for(size_t i=1;i<curve.size();++i)
      if(pwm<=curve[i].first)
      {
        double fraction=(pwm-curve[i-1].first)/(curve[i].first-curve[i-1].first);
        return curve[i-1].second+fraction*(curve[i].second-curve[i-1].second);
      }
    return curve.back().second;
  }

  void PWM(const gz::msgs::Double_V &msg)
  {
    std::lock_guard<std::mutex> guard(mutex);
    pwmMode=true;lastCommand=std::chrono::steady_clock::now();
    pwmLeft=pwmRight=0;wheelDuty.fill(0);fourDuty=msg.data_size()==4;
    if(msg.data_size()!=2 && msg.data_size()!=4) return;
    for(int i=0;i<msg.data_size();++i) if(!std::isfinite(msg.data(i))) return;
    if(fourDuty) for(int i=0;i<4;++i) wheelDuty[i]=std::clamp(msg.data(i),-1.,1.);
    else {
      pwmLeft=std::clamp(msg.data(0),-1.,1.);
      pwmRight=std::clamp(msg.data(1),-1.,1.);
      for(int i=0;i<4;++i) wheelDuty[i]=i%2==0?pwmLeft:pwmRight;
    }
  }

  void Command(const gz::msgs::Twist &msg)
  {
    std::lock_guard<std::mutex> guard(mutex);
    pwmMode=false;lastCommand=std::chrono::steady_clock::now();
    if (!std::isfinite(msg.linear().y()) || !std::isfinite(msg.linear().x()) || !std::isfinite(msg.angular().z()))
    { linear = lateral = angular = 0; return; }
    linear = std::clamp(msg.linear().x(), -.3, .3);
    lateral = std::clamp(msg.linear().y(), -.3, .3);
    angular = std::clamp(msg.angular().z(), -1., 1.);
  }

  public: void Reset(const gz::sim::UpdateInfo &,
      gz::sim::EntityComponentManager &ecm) override
  {
    std::lock_guard<std::mutex> guard(mutex);
    linear = lateral = angular = pwmLeft = pwmRight = 0;
    wheelDuty.fill(0);
    pwmMode=false;
    integral.fill(0);filteredDuty.fill(0);brakeEngagement.fill(0);
    SampleParameters();
    lastPub = {};
    for (auto j : joints)
      if (j != gz::sim::kNullEntity && ecm.HasEntity(j))
      {
        ecm.SetComponentData<gz::sim::components::JointForceCmd>(j, {0.});
        // Physics resets its velocity request bookkeeping. Recreate the
        // request next PreUpdate instead of retaining a stale zero component.
        ecm.RemoveComponent<gz::sim::components::JointVelocity>(j);
      }
    joints.fill(gz::sim::kNullEntity);
  }

  public: void PreUpdate(const gz::sim::UpdateInfo &info,
      gz::sim::EntityComponentManager &ecm) override
  {
    if (info.paused) return;
    double v,vy,w,pl,pr;bool direct,four,expired;std::array<double,4> duties;
    { std::lock_guard<std::mutex> guard(mutex); v=linear;vy=lateral;w=angular;pl=pwmLeft;pr=pwmRight;direct=pwmMode;four=fourDuty;duties=wheelDuty;
      expired=std::chrono::steady_clock::now()-lastCommand > std::chrono::milliseconds(400);
      if(expired)
        {v=vy=w=pl=pr=0;duties.fill(0);} }
    gz::msgs::Double_V output;
    for (size_t i=0; i<joints.size(); ++i)
    {
      if (joints[i] == gz::sim::kNullEntity || !ecm.HasEntity(joints[i]))
      {
        joints[i] = model.JointByName(ecm, names[i]);
        if (joints[i] == gz::sim::kNullEntity) return;
        ecm.CreateComponent(joints[i], gz::sim::components::JointVelocity());
        auto wheel=model.LinkByName(ecm,names[i].substr(0,names[i].size()-6));
        auto inertial=ecm.Component<gz::sim::components::Inertial>(wheel);
        if(inertial) wheelInertia[i]=inertial->Data().MassMatrix().Iyy();
      }
      auto velocity = ecm.Component<gz::sim::components::JointVelocity>(joints[i]);
      if(!velocity || velocity->Data().empty())
      {
        // Missing feedback is not zero measured speed. Never drive blind.
        ecm.SetComponentData<gz::sim::components::JointForceCmd>(joints[i], {0.});
        for(int k=0;k<4;++k) output.add_data(0.);
        continue;
      }
      double omega = velocity->Data()[0];
      double side = i % 2 == 0 ? -1 : 1;
      double hand=(i==0 || i==3)?1:-1;
      double target = std::clamp((v - (mecanum?hand*vy:0) + side*w*(track+(mecanum?wheelbase:0))/2)/radius, -omega0, omega0);
      // A velocity servo chooses effective voltage. The DC back-EMF relation
      // then produces finite output torque, decreasing with shaft speed.
      // At zero target, terminals are shorted (dynamic braking), not PID hold.
      double error = target-omega;
      double dt = std::clamp(std::chrono::duration<double>(info.dt).count(), 0., .01);
      if (target == 0) integral[i]=0;
      else
      {
        double voltage = target/omega0 + feedback*error + integral[i];
        if (std::abs(voltage) < 1 || error*voltage < 0)
          integral[i]=std::clamp(integral[i]+integralGain*error*dt, -1., 1.);
      }
      double duty = target == 0 ? 0 :
          std::clamp(target/omega0 + feedback*error + integral[i], -1., 1.);
      if(direct)
      {
        integral[i]=0;
        double effort=(mecanum || four)?-1:EffectiveDuty(pl,pr),magnitude=std::max(std::abs(pl),std::abs(pr));
        duty=effort>=0 ? (magnitude>1e-9 ? (i%2==0 ? pl : pr)/magnitude*effort : 0) :
            std::clamp((mecanum || four)?duties[i]:(i%2==0 ? pl*leftVoltageGain : pr*rightVoltageGain),-1.,1.);
        auto sign=[](double x){return int(x>1e-9)-int(x < -1e-9);};
        if(mecanum || four)
        {
          double magnitude=std::max(0.,(std::abs(duty)-wheelDeadzone[i])/(1.-wheelDeadzone[i]));
          duty=std::clamp(std::copysign(std::pow(magnitude,wheelExponent[i]),duty)*(duty<0?wheelReverseGain[i]:wheelGain[i])*sampledGain[i],-1.,1.);
        }
        double tau=(mecanum || four)?wheelDriveTau[i]:riseTau[(sign(pl+pr)+1)*3+sign(pr-pl)+1];
        // Legacy skid-steer fit describes spin-up only. Mecanum drive_tau
        // is an aggregate effort-response lag for continuous commands.
        // Zero input still enters braking immediately; watchdog bypasses lag.
        if(duty==0 || tau<=0 || (!(mecanum || four) && std::abs(duty)<std::abs(filteredDuty[i]))) filteredDuty[i]=duty;
        else filteredDuty[i]+=(1-std::exp(-dt/tau))*(duty-filteredDuty[i]);
        duty=filteredDuty[i];
        target=duty*omega0;
      }
      double torque = std::clamp(stall*(duty-omega/omega0), -stall, stall);
      bool braking=direct ? std::abs(duties[i])<1e-9 : target==0;
      if(!braking) brakeEngagement[i]=0;
      if(braking && dt>0)
      {
        // Real zero PWM is the H-bridge brake state, including the held inside
        // wheels of an arc. Model gearbox/brake drag with finite resistance.
        // The inertia cap prevents an explicit damping step reversing the wheel.
        double resistance=std::min({stall,brakeDamping*std::abs(omega)+brakeResistance,
                                   wheelInertia[i]*std::abs(omega)/dt});
        // Aggregate loaded brake take-up fitted to continuous W -> WD IMU
        // response, not a claimed electrical H-bridge delay. Full release and
        // watchdog expiry bypass this lag and brake all wheels immediately.
        double brakeTau=expired?0:(mecanum || four)?wheelBrakeTau[i]:
            ((direct && (std::abs(pl)+std::abs(pr)>1e-9))?steeringBrakeTau:0);
        if(brakeTau>0)
          brakeEngagement[i]+=(1-std::exp(-dt/brakeTau))*(1-brakeEngagement[i]);
        else brakeEngagement[i]=1;
        torque=-brakeEngagement[i]*std::copysign(resistance,omega);
      }
      if (!std::isfinite(torque)) torque = 0;
      ecm.SetComponentData<gz::sim::components::JointForceCmd>(joints[i], {torque});
      output.add_data(omega);
      output.add_data(target);
      output.add_data(torque);
      output.add_data(duty);
    }
    if (info.simTime-lastPub >= std::chrono::milliseconds(20))
    { telemetry.Publish(output); lastPub=info.simTime; }
  }
};
}
GZ_ADD_PLUGIN(ugv::TTMotorSystem, gz::sim::System,
    ugv::TTMotorSystem::ISystemConfigure, ugv::TTMotorSystem::ISystemPreUpdate,
    ugv::TTMotorSystem::ISystemReset)
GZ_ADD_PLUGIN_ALIAS(ugv::TTMotorSystem, "ugv::TTMotorSystem")
