// Grady Conwell, 2026
//
// Standing Wave: radiation forces on the jet and on detached drops. Physics and conventions: acousticWave.hpp.

#include "acousticForcing.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <heyoka/heyoka.hpp>

namespace {

constexpr double PI = std::numbers::pi;

// Volume average of sin(q z) over a sphere of radius R, relative to its value at the centre: 3 j1(x) / x with x = q R.
double sphereAverageOfSinusoid(double x) {
    if (x < 1e-3) return 1.0 - x * x / 10.0;
    return 3.0 * (std::sin(x) - x * std::cos(x)) / (x * x * x);
}

// Gor'kov contrast factor Phi = f1 / 3 + f2 / 2 of a liquid sphere in air.
// See https://en.wikipedia.org/wiki/Acoustic_contrast_factor
double gorkovContrastFactor(const AcousticWaveSettings& settings, double liquidDensity) {
    const double compressibilityRatio = (settings.airDensity * settings.airSoundSpeed * settings.airSoundSpeed) / (liquidDensity * settings.liquidSoundSpeed * settings.liquidSoundSpeed);
    const double monopoleCoeff = 1.0 - compressibilityRatio;       // f1
    const double dipoleCoeff = 2.0 * (liquidDensity - settings.airDensity) / (2.0 * liquidDensity + settings.airDensity);       // f2
    return monopoleCoeff / 3.0 + dipoleCoeff / 2.0;
}

// Distance ahead of the nearest pressure node is xi = z - nodePositionAtStart - patternVelocity t (mod nodeSpacing).
// Templated so the jet (heyoka expressions) and the drops (doubles) share one definition.
template <class Value>
Value distanceAheadOfNode(const AcousticWave& wave, const Value& z, const Value& t) {
    return z - wave.nodePositionAtStart - wave.patternVelocity * t;
}

} // namespace

AcousticWave makeAcousticWave(const AcousticWaveSettings& settings, double liquidDensity, double lengthScale, double timeScale) {
    const double nodeSpacing = settings.dropletSpacing / settings.pressureNodesPerDropletSpacing; // [m]
    const double wavenumber = PI / nodeSpacing;                                                   // [1/m]
    const double energyDensity = settings.pressureAmplitude * settings.pressureAmplitude / (4.0 * settings.airDensity * settings.airSoundSpeed * settings.airSoundSpeed);
    const double contrastFactor = gorkovContrastFactor(settings, liquidDensity);
    const double accelerationScale = lengthScale / (timeScale * timeScale); // [m/s^2]
    const double capillaryPressure = liquidDensity * accelerationScale * lengthScale; // gamma / h0 = rho h0^2 / tc^2

    AcousticWave wave;
    wave.actsOnJet = settings.actsOnJet;
    wave.actsOnDrops = settings.actsOnDrops;
    wave.wavenumber = wavenumber * lengthScale;
    wave.patternVelocity = settings.patternVelocity * timeScale / lengthScale;
    wave.nodePositionAtStart = settings.nodePositionAtStart / lengthScale;
    wave.jetPeakAcceleration = 2.0 * wavenumber * energyDensity / liquidDensity / accelerationScale;
    wave.dropPeakAcceleration = 3.0 * contrastFactor * wavenumber * energyDensity / liquidDensity / accelerationScale;

    wave.settings = settings;
    wave.nodeSpacing = nodeSpacing;
    wave.frequency = settings.airSoundSpeed / (2.0 * nodeSpacing);
    wave.energyDensity = energyDensity;
    wave.contrastFactor = contrastFactor;
    wave.soundPressureLevel = 20.0 * std::log10(settings.pressureAmplitude / std::sqrt(2.0) / 20e-6);
    wave.energyDensityOverCapillaryPressure = energyDensity / capillaryPressure;
    return wave;
}

heyoka::expression acousticAccelerationOnJet(const AcousticWave& wave, const heyoka::expression& z, const heyoka::expression& t) {
    return -wave.jetPeakAcceleration * heyoka::sin(2.0 * wave.wavenumber * distanceAheadOfNode(wave, z, t));
}

double acousticAccelerationOnDrop(const AcousticWave& wave, double z, double radius, double t) {
    const double volumeAverage = sphereAverageOfSinusoid(2.0 * wave.wavenumber * radius);
    return -wave.dropPeakAcceleration * volumeAverage * std::sin(2.0 * wave.wavenumber * distanceAheadOfNode(wave, z, t));
}

void advanceDropInAcousticWave(const AcousticWave& wave, double radius, double& position, double& velocity, double fromTime, double toTime) {
    constexpr double maxStep = 0.05; // tc; well below both the trap period (~70 tc) and the node-crossing time (> 1 tc)
    const int stepCount = std::max(1, int(std::ceil((toTime - fromTime) / maxStep)));
    const double dt = (toTime - fromTime) / stepCount;
    auto acceleration = [&](double z, double t) { return acousticAccelerationOnDrop(wave, z, radius, t); };
    for (int step = 0; step < stepCount; step++) {
        const double t = fromTime + step * dt, z = position, v = velocity;
        const double a1 = acceleration(z, t);
        const double a2 = acceleration(z + 0.5 * dt * v, t + 0.5 * dt);
        const double a3 = acceleration(z + 0.5 * dt * (v + 0.5 * dt * a1), t + 0.5 * dt);
        const double a4 = acceleration(z + dt * (v + 0.5 * dt * a2), t + dt);
        position = z + dt * (v + dt * (a1 + a2 + a3) / 6.0);
        velocity = v + dt * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0;
    }
}

std::string acousticWaveParameterRows(const AcousticWave& wave) {
    const auto& s = wave.settings;
    return std::format("acoustic_on_jet,{:d},\n"
                       "acoustic_on_drops,{:d},\n"
                       "acoustic_pressure,{},Pa\n"
                       "acoustic_frequency,{},MHz\n"
                       "acoustic_node_spacing,{},um\n"
                       "acoustic_node_velocity,{},m/s\n"
                       "acoustic_node_start,{},um\n",
                       s.actsOnJet, s.actsOnDrops, s.pressureAmplitude, wave.frequency * 1e-6, wave.nodeSpacing * 1e6,
                       s.patternVelocity, s.nodePositionAtStart * 1e6);
}

std::string describeAcousticWave(const AcousticWave& wave) {
    const auto& s = wave.settings;
    return std::format(
        "Acoustic standing wave: on jet = {}, on drops = {}\n"
        "  pA = {:.0f} Pa ({:.1f} dB SPL), f = {:.3f} MHz, node spacing = {:.1f} um ({} per {:.1f} um droplet spacing)\n"
        "  nodes move at {:.2f} m/s, one at z = {:.1f} um at t = 0\n"
        "  E = {:.2f} J/m^3 = {:.2f}% of gamma / h0, Gor'kov contrast factor Phi = {:.4f}\n",
        s.actsOnJet, s.actsOnDrops, s.pressureAmplitude, wave.soundPressureLevel, wave.frequency * 1e-6,
        wave.nodeSpacing * 1e6, s.pressureNodesPerDropletSpacing, s.dropletSpacing * 1e6, s.patternVelocity,
        s.nodePositionAtStart * 1e6, wave.energyDensity, 100.0 * wave.energyDensityOverCapillaryPressure,
        wave.contrastFactor);
}
