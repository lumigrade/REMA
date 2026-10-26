// REMA Antilatency "what the Alt sees" recorder.
//
//   alt_rays_bridge --serial SN [--track-seconds 4] [--rays-seconds 10] [--sdk-lib DIR] [--env-name NAME]
//
// Phase 1 (tracking task): prints {"event":"pose",...} = mean RAW pose (identity placement) of the
//   last second, so expected marker directions can be computed. The rig must be still.
// Phase 2 (tracking-data task; an Alt runs one task at a time, so no pose here): one JSON line per
//   optical frame: {"t_ns":..,"up":[x,y,z],"rays":[[mx,my,dx,dy,dz,power],...]}
//   mx,my = OpticRay.middleSpacePoint, d = OpticRay.direction, power = blob brightness.
// Also prints environment markers and the optics->body rotation. Exits when stdin closes or time ends.
#include <Antilatency.DeviceNetwork.h>
#include <Antilatency.Alt.Tracking.h>
#include <Antilatency.Alt.Environment.Selector.h>
#include <Antilatency.StorageClient.h>
#include <Antilatency.InterfaceContract.LibraryLoader.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace DN = Antilatency::DeviceNetwork;
namespace AT = Antilatency::Alt::Tracking;
namespace AE = Antilatency::Alt::Environment;
namespace IC = Antilatency::InterfaceContract;

static long long mono_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

template <typename T>
static T load(const std::string& dir, const std::string& name) {
    auto path = dir + "/lib" + name + ".so";
    T lib = IC::getLibraryInterface<T>(path.c_str());
    if (lib == nullptr) throw std::runtime_error("Cannot load " + path);
    return lib;
}

static std::mutex out_mutex;

class RayCallback : public IC::InterfacedObject {
    IMPLEMENT_INTERFACE(RayCallback, AT::ITrackingDataCallback)
public:
    std::atomic<long> frames{0};
    void onTrackingFrame(const std::vector<AT::OpticRay>& rays, Antilatency::Math::float3 up, AT::InertialLeap,
                         IC::LongBool accOverflow, IC::LongBool gyroOverflow) {
        const long long t = mono_ns();
        std::string s = "{\"t_ns\":" + std::to_string(t);
        char b[160];
        std::snprintf(b, sizeof b, ",\"up\":[%.5f,%.5f,%.5f],\"acc_overflow\":%d,\"gyro_overflow\":%d,\"rays\":[",
                      up.x, up.y, up.z, accOverflow ? 1 : 0, gyroOverflow ? 1 : 0);
        s += b;
        for (size_t i = 0; i < rays.size(); ++i) {
            const auto& r = rays[i];
            std::snprintf(b, sizeof b, "%s[%.6f,%.6f,%.6f,%.6f,%.6f,%.4f]", i ? "," : "", r.middleSpacePoint.x,
                          r.middleSpacePoint.y, r.direction.x, r.direction.y, r.direction.z, r.power);
            s += b;
        }
        s += "]}\n";
        std::lock_guard<std::mutex> lk(out_mutex);
        std::fputs(s.c_str(), stdout);
        ++frames;
    }
    void onIncompleteLeap(AT::InertialLeap) {}
    void onAdnFinalize() {}
};

int main(int argc, char** argv) {
    std::string serial, envName, libDir = "lib";
    double trackSeconds = 4, raySeconds = 10;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--serial") serial = next();
        else if (a == "--track-seconds") trackSeconds = std::stod(next());
        else if (a == "--rays-seconds") raySeconds = std::stod(next());
        else if (a == "--sdk-lib") libDir = next();
        else if (a == "--env-name") envName = next();
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    try {
        auto dnLib = load<DN::ILibrary>(libDir, "AntilatencyDeviceNetwork");
        auto altLib = load<AT::ILibrary>(libDir, "AntilatencyAltTracking");
        auto envLib = load<AE::Selector::ILibrary>(libDir, "AntilatencyAltEnvironmentSelector");
        auto storage = load<Antilatency::StorageClient::ILibrary>(libDir, "AntilatencyStorageClient").getLocalStorage();
        if (envName.empty()) envName = storage.read("environment", ".default");
        auto environment = envLib.createEnvironment(storage.read("environment", envName));
        auto filter = dnLib.createFilter();
        filter.addUsbDevice(DN::Constants::AllUsbDevices);
        auto net = dnLib.createNetwork(filter);
        {
            std::string m = "{\"event\":\"environment\",\"name\":\"" + envName + "\",\"markers\":[";
            auto ms = environment.getMarkers();
            char b[96];
            for (size_t i = 0; i < ms.size(); ++i) { std::snprintf(b, sizeof b, "%s[%.5f,%.5f,%.5f]", i ? "," : "", ms[i].x, ms[i].y, ms[i].z); m += b; }
            std::printf("%s]}\n", m.c_str());
        }
        auto findNode = [&](auto& ctor) {
            for (int k = 0; k < 100; ++k) {
                for (auto node : ctor.findSupportedNodes(net)) {
                    if (net.nodeGetStatus(node) != DN::NodeStatus::Idle) continue;
                    std::string sn;
                    try { sn = net.nodeGetStringProperty(node, "sys/HardwareSerialNumber"); } catch (...) {}
                    if (serial.empty() || sn == serial) return node;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            throw std::runtime_error("Alt not found or busy (close other trackers: AntilatencyService, alt2freed, VPcamtracking UI)");
        };

        // ---- phase 1: pose
        {
            auto ctor = altLib.createTrackingCotaskConstructor();
            auto node = findNode(ctor);
            auto task = ctor.startTask(net, node, environment);
            std::vector<AT::State> hist;
            const auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::steady_clock::now() - t0 < std::chrono::duration<double>(trackSeconds)) {
                auto st = task.getExtrapolatedState({{0, 0, 0}, {0, 0, 0, 1}}, 0.0f);
                hist.push_back(st);
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            size_t n6 = 0;
            double p[3] = {0, 0, 0}, q[4] = {0, 0, 0, 0};
            const size_t from = hist.size() > 200 ? hist.size() - 200 : 0;
            Antilatency::Math::floatQ ref = hist.back().pose.rotation;
            for (size_t i = from; i < hist.size(); ++i) {
                const auto& s = hist[i];
                if (s.stability.stage != AT::Stage::Tracking6Dof) continue;
                ++n6;
                p[0] += s.pose.position.x; p[1] += s.pose.position.y; p[2] += s.pose.position.z;
                const auto& r = s.pose.rotation;
                const double sg = (r.x * ref.x + r.y * ref.y + r.z * ref.z + r.w * ref.w) < 0 ? -1 : 1;
                q[0] += sg * r.x; q[1] += sg * r.y; q[2] += sg * r.z; q[3] += sg * r.w;
            }
            if (n6) {
                const double qn = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
                std::printf("{\"event\":\"pose\",\"samples_6dof\":%zu,\"p\":[%.6f,%.6f,%.6f],\"q\":[%.7f,%.7f,%.7f,%.7f],\"quality\":%.4f}\n",
                            n6, p[0] / n6, p[1] / n6, p[2] / n6, q[0] / qn, q[1] / qn, q[2] / qn, q[3] / qn, hist.back().stability.value);
            } else {
                std::printf("{\"event\":\"pose\",\"samples_6dof\":0,\"stage\":\"%s\"}\n", Antilatency::enumToString(hist.back().stability.stage));
            }
            std::fflush(stdout);
            task = {};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // ---- phase 2: rays
        auto dctor = altLib.createTrackingDataCotaskConstructor();
        auto node = findNode(dctor);
        auto cb = IC::ObjectPtr<RayCallback>::create();
        AT::ITrackingDataCallback cbi = cb.queryInterface<AT::ITrackingDataCallback>();
        auto dtask = dctor.startTask(net, node, cbi);
        auto o2b = dtask.getOpticsToBodyRotation();
        {
            std::lock_guard<std::mutex> lk(out_mutex);
            std::printf("{\"event\":\"optics_to_body\",\"q\":[%.7f,%.7f,%.7f,%.7f]}\n", o2b.x, o2b.y, o2b.z, o2b.w);
            std::fflush(stdout);
        }
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - t0 < std::chrono::duration<double>(raySeconds) && !dtask.isTaskFinished()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            std::lock_guard<std::mutex> lk(out_mutex);
            std::fflush(stdout);
        }
        dtask = {};
        std::lock_guard<std::mutex> lk(out_mutex);
        std::printf("{\"event\":\"done\",\"ray_frames\":%ld}\n", cb->frames.load());
    } catch (const std::exception& e) {
        std::printf("{\"event\":\"error\",\"message\":\"%s\"}\n", e.what());
        return 1;
    }
    return 0;
}
