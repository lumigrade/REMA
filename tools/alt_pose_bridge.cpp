// REMA Antilatency pose bridge: raw Alt tracker pose as JSON lines with CLOCK_MONOTONIC stamps.
//
//   alt_pose_bridge [--rate 250] [--serial SN] [--env-name NAME | --env CODE] [--sdk-lib DIR]
//
// One line per sample: {"t_ns":..,"p":[x,y,z],"q":[x,y,z,w],"v":[..],"w":[..],"stage":"..","quality":..}
// p/q are the RAW tracker pose (identity placement, zero extrapolation) in Antilatency's
// left-handed frame; REMA applies its own calibrated Alt->camera transform. t_ns is the host
// CLOCK_MONOTONIC time of the SDK query, same clock as V4L2 kernel timestamps.
// Exits when stdin closes (so the parent owns its lifetime). Derived from VPcamtracking alt2freed.
#include <Antilatency.DeviceNetwork.h>
#include <Antilatency.Alt.Tracking.h>
#include <Antilatency.Alt.Environment.Selector.h>
#include <Antilatency.StorageClient.h>
#include <Antilatency.InterfaceContract.LibraryLoader.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <string>
#include <thread>

namespace DN = Antilatency::DeviceNetwork;
namespace AT = Antilatency::Alt::Tracking;
namespace AE = Antilatency::Alt::Environment;
namespace IC = Antilatency::InterfaceContract;

static std::atomic<bool> running{true};

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

static std::string prop(DN::INetwork& net, DN::NodeHandle node, const char* key) {
    try { return net.nodeGetStringProperty(node, key); } catch (...) { return ""; }
}

int main(int argc, char** argv) {
    std::signal(SIGINT, [](int) { running = false; });
    std::signal(SIGTERM, [](int) { running = false; });
    double rate = 250;
    std::string serial, envName, envCode, libDir = "lib";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--rate") rate = std::stod(next());
        else if (a == "--serial") serial = next();
        else if (a == "--env-name") envName = next();
        else if (a == "--env") envCode = next();
        else if (a == "--sdk-lib") libDir = next();
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    try {
        auto dnLib = load<DN::ILibrary>(libDir, "AntilatencyDeviceNetwork");
        auto altLib = load<AT::ILibrary>(libDir, "AntilatencyAltTracking");
        auto envLib = load<AE::Selector::ILibrary>(libDir, "AntilatencyAltEnvironmentSelector");
        if (envCode.empty()) {
            auto storage = load<Antilatency::StorageClient::ILibrary>(libDir, "AntilatencyStorageClient").getLocalStorage();
            if (envName.empty()) envName = storage.read("environment", ".default");
            envCode = storage.read("environment", envName);
        }
        if (envCode.empty()) throw std::runtime_error("no environment (set a default in AntilatencyService or pass --env-name)");
        auto environment = envLib.createEnvironment(envCode);
        auto filter = dnLib.createFilter();
        filter.addUsbDevice(DN::Constants::AllUsbDevices);
        auto net = dnLib.createNetwork(filter);
        auto ctor = altLib.createTrackingCotaskConstructor();
        std::printf("{\"event\":\"started\",\"sdk_device_network\":\"%s\",\"environment\":\"%s\",\"rate\":%.1f}\n",
                    dnLib.getVersion().c_str(), envName.c_str(), rate);
        std::fflush(stdout);
        std::thread([] { std::string l; while (std::getline(std::cin, l)) {} running = false; }).detach();

        AT::ITrackingCotask task;
        uint32_t lastUpdate = UINT32_MAX;
        auto nextScan = std::chrono::steady_clock::now();
        const Antilatency::Math::floatP3Q identity{{0, 0, 0}, {0, 0, 0, 1}};
        const auto period = std::chrono::nanoseconds((long long)(1e9 / rate));
        auto next = std::chrono::steady_clock::now();
        while (running) {
            next += period;
            if (task != nullptr && task.isTaskFinished()) {
                // Log when and how the SDK ended the task, then rescan straight away: waiting for the network
                // update id to change cost ~8 s per dropout when the node was still present.
                int present = 0, idle = 0;
                for (auto node : ctor.findSupportedNodes(net)) {
                    if (!serial.empty() && prop(net, node, "sys/HardwareSerialNumber") != serial) continue;
                    ++present;
                    if (net.nodeGetStatus(node) == DN::NodeStatus::Idle) ++idle;
                }
                std::printf("{\"event\":\"disconnected\",\"t_ns\":%lld,\"node_present\":%d,\"node_idle\":%d}\n", mono_ns(), present, idle);
                task = {};
                lastUpdate = UINT32_MAX;
            }
            auto upd = net.getUpdateId();
            const auto now = std::chrono::steady_clock::now();
            if (task == nullptr && (upd != lastUpdate || now >= nextScan)) {
                lastUpdate = upd;
                nextScan = now + std::chrono::milliseconds(100);
                for (auto node : ctor.findSupportedNodes(net)) {
                    if (net.nodeGetStatus(node) != DN::NodeStatus::Idle) continue;
                    auto sn = prop(net, node, "sys/HardwareSerialNumber");
                    if (!serial.empty() && sn != serial) continue;
                    task = ctor.startTask(net, node, environment);
                    std::printf("{\"event\":\"tracking\",\"t_ns\":%lld,\"serial\":\"%s\",\"firmware\":\"%s\"}\n", mono_ns(), sn.c_str(),
                                prop(net, node, "sys/FirmwareVersion").c_str());
                    break;
                }
            }
            if (task != nullptr) {
                auto s = task.getExtrapolatedState(identity, 0.0f);
                const long long t = mono_ns();
                const auto& p = s.pose.position;
                const auto& q = s.pose.rotation;
                std::printf("{\"t_ns\":%lld,\"p\":[%.6f,%.6f,%.6f],\"q\":[%.7f,%.7f,%.7f,%.7f],"
                            "\"v\":[%.5f,%.5f,%.5f],\"w\":[%.5f,%.5f,%.5f],\"stage\":\"%s\",\"quality\":%.4f}\n",
                            t, p.x, p.y, p.z, q.x, q.y, q.z, q.w, s.velocity.x, s.velocity.y, s.velocity.z,
                            s.localAngularVelocity.x, s.localAngularVelocity.y, s.localAngularVelocity.z,
                            Antilatency::enumToString(s.stability.stage), s.stability.value);
            }
            std::fflush(stdout);
            std::this_thread::sleep_until(next);
            if (std::chrono::steady_clock::now() - next > period * 4) next = std::chrono::steady_clock::now();
        }
    } catch (const std::exception& e) {
        std::printf("{\"event\":\"error\",\"message\":\"%s\"}\n", e.what());
        return 1;
    }
    return 0;
}
