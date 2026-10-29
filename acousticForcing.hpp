// Grady Conwell, 2026
//
// acoustic forcing of Rayleigh breakup.
//
// A longitudinal standing wave in the air, p1 = pA cos(theta) cos(w_ac t), lies along the jet axis. Its pressure nodes
// (cos(theta) = 0) are half an acoustic wavelength apart, and the whole pattern translates along the axis at
// patternVelocity, so a drop moving at that velocity sits still relative to it. The acoustic period (~1 µs) is short
// compared with the capillary time (~10 µs), so only the time-averaged (second-order) radiation forces are modeled:
//
// E   = pA^2 / (4 rho_air c_air^2)                     acoustic energy density
// k   = pi / nodeSpacing                               acoustic wavenumber
// xi  = z - nodePositionAtStart - patternVelocity t    distance ahead of a pressure node (mod nodeSpacing)
//
// Jet (droplet formation). The radiation pressure on the jet surface,
//
// p_rad = <p1^2> / (2 rho_air c_air^2) - rho_air <u1^2> / 2 = -E cos(2 k xi),
//
// is transmitted across the slender cross-section like the capillary pressure, so it acts as a potential per unit
//
// volume:  acceleration = -(1 / rho) dp_rad/dz = -(2 k E / rho) sin(2 k xi).
//
// Drops (after pinch-off). The Gor'kov potential of a sphere, U = V E [f1 cos^2(theta) - (3/2) f2 sin^2(theta)], gives
//
// acceleration = -(3 Phi k E / rho) sin(2 k xi) * 3 j1(2 k R) / (2 k R),    Phi = f1 / 3 + f2 / 2,
//
// with f1 = 1 - kappa_liquid / kappa_air and f2 = 2 (rho - rho_air) / (2 rho + rho_air). The last factor averages
//
// the force over the drop volume; it is a first finite-size correction, since Gor'kov assumes R << wavelength.
//
// Both accelerations point to the nearest pressure node, so the nodes are potential wells: liquid gathers at them and
// drops are held in them. No air drag is modeled, so a trapped drop oscillates about its node without damping.
//
// The breakup model (main.cpp) calls this module in exactly two places, each marked "ACOUSTIC FORCING":
// buildJetIntegrator() for the jet and advanceDrop() for detached drops.

#pragma once

#include <string>
#include <heyoka/expression.hpp>

// The wave as it would be set up in the lab, in SI units. Independent of the breakup model.
// Simulation is based on some arbitrary waveform, and neglects any feedback, coupling, 
// attenuation, and beam width effects. 
struct AcousticWaveSettings {
    bool actsOnJet;                     // apply the radiation pressure to the slender jet (droplet formation)
    bool actsOnDrops;                   // apply the radiation force to detached drops
    double pressureAmplitude;           // [Pa]   peak acoustic pressure pA, at a pressure antinode
    double dropletSpacing;              // [m]    drop-to-drop distance the wave is tuned to
    int pressureNodesPerDropletSpacing; //        pressure nodes (potential wells) per droplet spacing; tunes the pitch
    double patternVelocity;             // [m/s]  velocity of the nodes along the jet axis (set to the drop velocity) (probably requires one simulation with no acoustic forcing)
    double nodePositionAtStart;         // [m]    position of one pressure node at t = 0
    double airDensity;                  // [kg/m^3]
    double airSoundSpeed;               // [m/s]
    double liquidSoundSpeed;            // [m/s]  only sets the compressibility contrast f1 (~1 for any liquid in air)
};

// Demo values for the Van Hoeve et al. (2010) jet (h0 = 18.5 µm, 5.4 m/s, forcing period 30.43 µs).
// 
// Unforced model makes primary drops at 4.78 m/s (paper: 4.68 model, 4.9 experiment), one per forcing period,
// so they are 4.78 m/s * 30.43 µs = 145.4 µm apart. Acoustic frequency: 343.2 / (2 * 145.4 µm) = 1.18 MHz.
// 
// Phase: the unforced drop centres pass through z = 4.78 m/s * t (mod 145.4 µm).

inline const AcousticWaveSettings demoAcousticWave{
    .actsOnJet = false,
    .actsOnDrops = false,
    .pressureAmplitude = 5000.0, // 165 dB SPL; within reach of ultrasonic levitators. E = 44 J/m^3, 1.2% of gamma / h0
    .dropletSpacing = 145.4e-6,
    .pressureNodesPerDropletSpacing = 1,
    .patternVelocity = 4.78,
    .nodePositionAtStart = 72.7e-6,
    .airDensity = 1.204,         // 20 °C
    .airSoundSpeed = 343.2,      // 20 °C
    .liquidSoundSpeed = 1680.0,  // 40% w/w glycerol-water
};

// The wave in model units (length h0, time tc), ready for the breakup model.
struct AcousticWave {
    bool actsOnJet, actsOnDrops;
    double wavenumber;           // k = pi / nodeSpacing                                  [1/h0]
    double patternVelocity;      //                                                       [h0/tc]
    double nodePositionAtStart;  //                                                       [h0]
    double jetPeakAcceleration;  // 2 k E / rho                                           [h0/tc^2]
    double dropPeakAcceleration; // 3 Phi k E / rho, for a point-like drop                 [h0/tc^2]

    // SI values, for reporting only
    AcousticWaveSettings settings;
    double nodeSpacing, frequency, energyDensity, contrastFactor, soundPressureLevel, energyDensityOverCapillaryPressure;
};

// Converts the lab settings to model units. lengthScale [m] and timeScale [s] are the model's h0 and tc.
AcousticWave makeAcousticWave(const AcousticWaveSettings& settings, double liquidDensity, double lengthScale,
                              double timeScale);

// Acceleration of the jet liquid at axial position z and time t (heyoka expressions, model units).
heyoka::expression acousticAccelerationOnJet(const AcousticWave& wave, const heyoka::expression& z,
                                             const heyoka::expression& t);

// Acceleration of a detached drop of the given radius, centred at z at time t (model units).
double acousticAccelerationOnDrop(const AcousticWave& wave, double z, double radius, double t);

// Moves a drop from fromTime to toTime under the acoustic force (classical RK4; model units).
void advanceDropInAcousticWave(const AcousticWave& wave, double radius, double& position, double& velocity,
                               double fromTime, double toTime);

// Multi-line summary of the wave in SI units.
std::string describeAcousticWave(const AcousticWave& wave);

// The wave as rows of "quantity,value,unit" (no header), appended to a run's parameters.csv.
std::string acousticWaveParameterRows(const AcousticWave& wave);
