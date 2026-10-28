// Grady Conwell, 2026
//
// Rayleigh breakup of a 18.5 µm microjet, after Van Hoeve et al. (2010), "Breakup of diminutive Rayleigh jets",
// using the slender-jet (lubrication) equations of Eggers & Dupont, J. Fluid Mech. 262, 205 (1994).
//
// Lengths are in h0, time in the capillary time tc = sqrt(rho h0^3 / gamma), velocities in uc = h0 / tc.
// The jet is written in terms of a = h^2, which keeps the equations regular at a rounded tip (h -> 0):
//   a_t + (a v)' = 0
//   (a v)_t + (a v^2)' = G' + 3 Oh (a v')'
//
// G is the axial surface-tension force on a cross-section (divided by pi gamma h0), so both equations conserve mass and
// momentum exactly.
//
// The N cells span the jet evenly from the nozzle (z = 0) to the tip (z = L), so z = xi L(t). Cell centres hold a,
// cell faces hold v (staggered: face 0 is the nozzle, face N the tip), and the tip moves with the fluid: dL/dt = v_N.
// Pinch-off: when the neck radius drops below 100 nm, the part downstream becomes a drop and the remaining jet is
// redistributed over the N cells. Detached drops are carried as (volume, momentum). Objects that touch
// (gap < 0.005 h0) while approaching coalesce. A drop caught by the jet tip is absorbed into the tip cells.
//
// SYMBOLS referenced in equations and comments (all dimensionless)
//   a, h       Squared radius h^2 and radius of the jet; a(-1) and a(N) are ghost cells for the nozzle and the tip
//   v          Axial velocity at a face
//   L, dz      Jet length (nozzle to tip) and cell width L / N
//   '          Derivative along z
//   Oh         Ohnesorge number mu / sqrt(rho gamma h0): viscosity vs. inertia and surface tension (~0.10)
//   U          Jet velocity at the nozzle, vjet / uc = sqrt(We) (~2.95)
//   G, S       Capillary tension h^2 K and viscous tension a v' on a cross-section
//
// OUTPUT  results/<YYYYMMDD_HHMMSS>/
//   jet.csv       t, L and h at every cell centre, one row per 1 µs frame
//   drops.csv     t, z, R, v of every drop in flight, per frame
//   events.csv    every pinch-off and absorption
//   neck.csv      minimum neck radius after every integrator step (for the Fig. 6 power law)
//   acoustic.txt  the acoustic wave used for the run
//   parameters.csv  model and acoustic inputs as quantity,value,unit (read by plotJetProfile.py)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <format> // and let there be python style text formatting.
#include <fstream>
#include <iostream>
#include <numbers>
#include <vector>
#include <heyoka/heyoka.hpp> // ODE library

#include "acousticForcing.hpp"

using namespace std;
using namespace heyoka;
constexpr double PI = numbers::pi;

// Physical inputs (SI) defaults for: 40% w/w glycerol-saline microjet, Van Hoeve et al. (2010) Sec. III.A
//const double liquidDensity = 1098.0;   // kg/m^3
//const double surfaceTension = 0.0679;  // N/m
//const double viscosity = 3.65e-3;      // Pa s
//const double jetRadius = 18.5e-6;      // m, unperturbed radius h0: the length scale
//const double jetVelocity = 5.4;        // m/s, at the nozzle
//const double nozzleModulation = 0.005; // amplitude of the nozzle radius modulation, as a fraction of h0 (paper: 0.005)

const double liquidDensity = 1098.0;   // kg/m^3
const double surfaceTension = 0.0679;  // N/m
const double viscosity = 3.65e-3;      // Pa s
const double jetRadius = 18.5e-6;      // m, unperturbed radius h0: the length scale
const double jetVelocity = 5.4;        // m/s, at the nozzle
const double nozzleModulation = 0.005; // amplitude of the nozzle radius modulation, as a fraction of h0 (paper: 0.005)

// Scales and dimensionless groups
const double capillaryTime = sqrt(liquidDensity * jetRadius * jetRadius * jetRadius / surfaceTension); // s, ~10 µs
const double velocityScale = jetRadius / capillaryTime;                                // m/s
const double ohnesorgeNumber = viscosity / sqrt(liquidDensity * surfaceTension * jetRadius);
const double meanNozzleVelocity = jetVelocity / velocityScale; // U = sqrt(We)
// 2 pi f with f = vjet / lambda_opt and lambda_opt = 2 sqrt(2) pi h0, the fastest-growing Rayleigh wavelength
const double forcingAngularFrequency = meanNozzleVelocity / sqrt(2.0);

// Numerical settings
const int cellCount = 1000;                  // cells from nozzle to tip, as in the paper
const double pinchRadius = 100e-9 / jetRadius; // neck radius at which the jet pinches off, 100 nm
const double coalescenceGap = 0.005;         // gap below which approaching objects coalesce
const double frameInterval = 1e-6 / capillaryTime; // 1 µs between output frames
double endTime = 400e-6 / capillaryTime;     // override with ./microjet <end time in µs>

// Conversion of model units to output units
const double toMicrometres = jetRadius * 1e6, toMicroseconds = capillaryTime * 1e6, toMetresPerSecond = velocityScale;

// State vector
// a at the cell centres 0 .. N-1, then v at the faces 1 .. N (face N is the tip; face 0, the nozzle, is not stored),
// then the jet length L.
constexpr int stateSize = 2 * cellCount + 1;
constexpr int squaredRadiusIndex(int cell) { return cell; }
constexpr int faceVelocityIndex(int face) { return cellCount + face - 1; }
constexpr int jetLengthIndex = 2 * cellCount;

// Named access to the integrator's numerical state.
struct JetState {
    double* values;
    double& squaredRadius(int cell) const { return values[squaredRadiusIndex(cell)]; }
    double& faceVelocity(int face) const { return values[faceVelocityIndex(face)]; }
    double& length() const { return values[jetLengthIndex]; }
    double tipVelocity() const { return faceVelocity(cellCount); }
    double cellWidth() const { return length() / cellCount; }
};

// A detached drop: volume, momentum (volume * velocity) and centre of mass `position` at `time`.
struct Drop {
    double volume, momentum, position, time;
    double velocity() const { return momentum / volume; }
    double radius() const { return cbrt(3.0 * volume / (4.0 * PI)); }
    double rearEdge() const { return position - radius(); }
    double frontEdge() const { return position + radius(); }
};

// Nozzle: h = 1 + delta sin(omega t) with a constant flow rate, so v = U / h^2 (paper, Eqs. 8-9).
// Templated so the integrator (heyoka expressions) and the grid remapping (doubles) share one definition.
template <class Time>
Time nozzleSquaredRadius(const Time& time) {
    auto nozzleRadius = 1.0 + nozzleModulation * sin(forcingAngularFrequency * time);
    return nozzleRadius * nozzleRadius;
}

// Slender-jet equations
// Capillary tension G = h^2 K = a^2 (1 + a''/2) / (a + a'^2/4)^(3/2) at a cell, from a at the cell and its neighbours.
expression capillaryTension(const expression& aBehind, const expression& aHere, const expression& aAhead,
                            const expression& dz) {
    auto slope = (aAhead - aBehind) / (2.0 * dz);                     // a'
    auto secondDerivative = (aAhead - 2.0 * aHere + aBehind) / (dz * dz); // a''
    auto curvatureDenominator = aHere + 0.25 * slope * slope;
    return aHere * aHere * (1.0 + 0.5 * secondDerivative) / (curvatureDenominator * sqrt(curvatureDenominator));
}

taylor_adaptive<double> buildJetIntegrator(vector<double> initialState, const AcousticWave& acousticWave) {
    vector<expression> x(stateSize);
    for (int i = 0; i < stateSize; i++) x[i] = make_vars("x" + to_string(i));

    const auto nozzleA = nozzleSquaredRadius(heyoka::time);
    const auto nozzleV = meanNozzleVelocity / nozzleA;
    const auto length = x[jetLengthIndex], tipVelocity = x[faceVelocityIndex(cellCount)];
    const auto dz = length / double(cellCount);
    // Ghost cells pin a = a_nozzle at z = 0 and a = 0 at the tip
    auto a = [&](int cell) {
        return cell < 0 ? 2.0 * nozzleA - x[0] : cell >= cellCount ? -x[cellCount - 1] : x[squaredRadiusIndex(cell)];
    };
    auto v = [&](int face) { return face == 0 ? nozzleV : x[faceVelocityIndex(face)]; };
    auto gridVelocity = [&](int face) { return (double(face) / cellCount) * tipVelocity; }; // grid stretches with L
    auto facePosition = [&](int face) { return (double(face) / cellCount) * length; };

    vector<expression> capillary(cellCount), viscous(cellCount), massFlux(cellCount + 1);
    for (int cell = 0; cell < cellCount; cell++) {
        capillary[cell] = capillaryTension(a(cell - 1), a(cell), a(cell + 1), dz);
        viscous[cell] = a(cell) * (v(cell + 1) - v(cell)) / dz; // viscous tension / 3 Oh
    }
    massFlux[0] = nozzleA * nozzleV;          // inflow
    massFlux[cellCount] = expression(0.0);    // the tip moves with the fluid
    for (int face = 1; face < cellCount; face++) // relative to the moving grid
        massFlux[face] = 0.5 * (a(face - 1) + a(face)) * (v(face) - gridVelocity(face));

    // Acceleration of the liquid at each face: capillary + viscous force per unit mass, minus advection
    vector<expression> faceAcceleration(cellCount + 1);
    for (int face = 1; face < cellCount; face++) {
        auto advection = (v(face) - gridVelocity(face)) * (v(face + 1) - v(face - 1)) / (2.0 * dz);
        auto force = (capillary[face] - capillary[face - 1] + 3.0 * ohnesorgeNumber * (viscous[face] - viscous[face - 1])) / (0.5 * (a(face - 1) + a(face)) * dz);
        faceAcceleration[face] = force - advection;
    }
    // Tip: the last half cell is pulled back by the tension of its neighbour (zero area, zero force at the tip)
    faceAcceleration[cellCount] = -(capillary[cellCount - 1] + 3.0 * ohnesorgeNumber * viscous[cellCount - 1]) / (0.5 * a(cellCount - 1) * dz);

    // Acoustic Forcing: radiation pressure on the jet (droplet formation) ================================
    if (acousticWave.actsOnJet)
        for (int face = 1; face <= cellCount; face++)
            faceAcceleration[face] = faceAcceleration[face]
                                   + acousticAccelerationOnJet(acousticWave, facePosition(face), heyoka::time);
    // ==========================================================================================================

    vector<pair<expression, expression>> odes;
    for (int cell = 0; cell < cellCount; cell++) { // mass from net inflow, minus dilution as the grid stretches with L
        auto stretching = a(cell) * tipVelocity / length;
        odes.push_back({x[squaredRadiusIndex(cell)], -(massFlux[cell + 1] - massFlux[cell]) / dz - stretching});
    }
    for (int face = 1; face <= cellCount; face++) odes.push_back({x[faceVelocityIndex(face)], faceAcceleration[face]});
    odes.push_back({length, tipVelocity});

    // Terminal events stop the integrator exactly when a cell reaches the pinch-off radius
    vector<taylor_adaptive<double>::t_event_t> pinchEvents;
    for (int cell = 3; cell < cellCount - 3; cell++)
        pinchEvents.emplace_back(x[squaredRadiusIndex(cell)] - pinchRadius * pinchRadius,
                                 kw::direction = event_direction::negative);
    return taylor_adaptive<double>{odes, std::move(initialState), kw::tol = 1e-10, kw::compact_mode = true,
                                   kw::t_events = pinchEvents};
}

// Hemispherical drop h = sqrt(h0^2 - z^2) at the nozzle, moving at vjet (paper, Sec. II).
vector<double> hemisphericalInitialState() {
    vector<double> state(stateSize, meanNozzleVelocity);
    for (int cell = 0; cell < cellCount; cell++) state[squaredRadiusIndex(cell)] = 1.0 - pow((cell + 0.5) / cellCount, 2);
    state[jetLengthIndex] = 1.0;
    return state;
}

bool isPhysical(JetState jet) {
    const bool allFinite = all_of(jet.values, jet.values + stateSize, [](double value) { return isfinite(value); });
    return allFinite && *min_element(jet.values, jet.values + cellCount) > 0.0;
}

// Pinch-off, absorption and coalescence
// Redistributes the cells over the jet [0, newLength] by linear interpolation of the old grid.
void remapGrid(JetState jet, double time, double newLength) {
    const double oldLength = jet.length(), dz = jet.cellWidth();
    vector<double> a(jet.values, jet.values + cellCount), v(cellCount + 1);
    v[0] = meanNozzleVelocity / nozzleSquaredRadius(time);
    for (int face = 1; face <= cellCount; face++) v[face] = jet.faceVelocity(face);

    auto squaredRadiusAt = [&](double z) { // from the cell centres, with a = 0 at the tip
        double fractionalCell = z / dz - 0.5;
        if (fractionalCell <= 0) return a[0];
        if (fractionalCell >= cellCount - 1) return max(0.0, a[cellCount - 1] * (oldLength - z) / (0.5 * dz));
        int cell = int(fractionalCell);
        return a[cell] + (fractionalCell - cell) * (a[cell + 1] - a[cell]);
    };
    auto velocityAt = [&](double z) { // from the faces
        double fractionalFace = min(z, oldLength) / dz;
        int face = min(int(fractionalFace), cellCount - 1);
        return v[face] + (fractionalFace - face) * (v[face + 1] - v[face]);
    };
    for (int cell = 0; cell < cellCount; cell++)
        jet.squaredRadius(cell) = squaredRadiusAt((cell + 0.5) * newLength / cellCount);
    for (int face = 1; face <= cellCount; face++) jet.faceVelocity(face) = velocityAt(face * newLength / cellCount);
    jet.length() = newLength;
}

// The thinnest local minimum of h below maxNeckRadius that has a real bulge (a drop) downstream of it; -1 if none.
int findNeck(JetState jet, double maxNeckRadius) {
    int thinnestNeck = -1;
    double largestDownstream = 0.0; // largest a downstream of the cell being tested
    for (int cell = cellCount - 1; cell > 2; cell--) {
        const double a = jet.squaredRadius(cell);
        largestDownstream = max(largestDownstream, a);
        if (cell >= cellCount - 3 || a >= maxNeckRadius * maxNeckRadius) continue;
        const bool isLocalMinimum = a <= jet.squaredRadius(cell - 1) && a <= jet.squaredRadius(cell + 1);
        const bool hasBulgeDownstream = largestDownstream > 100.0 * pinchRadius * pinchRadius;
        if (isLocalMinimum && hasBulgeDownstream && (thinnestNeck < 0 || a < jet.squaredRadius(thinnestNeck)))
            thinnestNeck = cell;
    }
    return thinnestNeck;
}

// The cells downstream of the neck become a drop, and the jet is shortened to end at the neck.
Drop pinchOff(JetState jet, int neckCell, double time) {
    const double dz = jet.cellWidth();
    Drop drop{0.0, 0.0, 0.0, time};
    for (int cell = neckCell + 1; cell < cellCount; cell++) {
        const double cellVolume = PI * jet.squaredRadius(cell) * dz;
        drop.volume += cellVolume;
        drop.momentum += cellVolume * 0.5 * (jet.faceVelocity(cell) + jet.faceVelocity(cell + 1));
        drop.position += cellVolume * (cell + 0.5) * dz;
    }
    drop.position /= drop.volume;
    remapGrid(jet, time, (neckCell + 0.5) * dz);
    return drop;
}

// Index of a slower drop the jet tip has reached; -1 if none.
int findDropCaughtByTip(JetState jet, const vector<Drop>& drops) {
    for (size_t i = 0; i < drops.size(); i++)
        if (drops[i].rearEdge() - jet.length() < coalescenceGap && jet.tipVelocity() > drops[i].velocity()) return int(i);
    return -1;
}

// Adds the drop's mass and momentum to the tip cells it overlaps.
void absorbDropIntoTip(JetState jet, const Drop& drop) {
    const double dz = jet.cellWidth();
    const int receivingCells = clamp(int(2.0 * drop.radius() / dz), 1, cellCount - 1);
    const double addedSquaredRadius = drop.volume / (PI * receivingCells * dz);
    const int firstCell = cellCount - receivingCells;
    for (int cell = firstCell; cell < cellCount; cell++) jet.squaredRadius(cell) += addedSquaredRadius;
    for (int face = firstCell; face <= cellCount; face++) { // mass-weighted face velocity
        const double oldSquaredRadius = jet.squaredRadius(min(face, cellCount - 1)) - addedSquaredRadius;
        jet.faceVelocity(face) = (oldSquaredRadius * jet.faceVelocity(face) + addedSquaredRadius * drop.velocity())
                               / (oldSquaredRadius + addedSquaredRadius);
    }
}

// Merges touching drops that are approaching each other (conserves volume and momentum). All drops must be at one time.
void coalesceTouchingDrops(vector<Drop>& drops) {
    sort(drops.begin(), drops.end(), [](const Drop& p, const Drop& q) { return p.position < q.position; });
    for (size_t i = 1; i < drops.size();) {
        Drop &behind = drops[i - 1], &ahead = drops[i];
        if (ahead.rearEdge() - behind.frontEdge() < coalescenceGap && behind.velocity() > ahead.velocity()) {
            const double volume = behind.volume + ahead.volume;
            behind = {volume, behind.momentum + ahead.momentum,
                      (behind.volume * behind.position + ahead.volume * ahead.position) / volume, behind.time};
            drops.erase(drops.begin() + i);
        } else i++;
    }
}

// Drop flight

void advanceDrop(Drop& drop, double toTime, const AcousticWave& acousticWave) {
    if (toTime <= drop.time) return;
    // Acoustic Forcing: radiation force on a detached drop
    if (acousticWave.actsOnDrops) {
        double velocity = drop.velocity();
        advanceDropInAcousticWave(acousticWave, drop.radius(), drop.position, velocity, drop.time, toTime);
        drop.momentum = drop.volume * velocity;
        drop.time = toTime;
        return;
    }
    // Free flight
    drop.position += drop.velocity() * (toTime - drop.time);
    drop.time = toTime;
}

void advanceDrops(vector<Drop>& drops, double toTime, const AcousticWave& acousticWave) {
    for (Drop& drop : drops) advanceDrop(drop, toTime, acousticWave);
}

// Output

filesystem::path createRunDirectory() {
    char stamp[32];
    const time_t now = std::time(nullptr);
    strftime(stamp, sizeof stamp, "%Y%m%d_%H%M%S", localtime(&now));
    auto directory = filesystem::path("results") / stamp;
    filesystem::create_directories(directory);
    return directory;
}

struct OutputFiles {
    ofstream jet, drops, events, neck;

    explicit OutputFiles(const filesystem::path& directory)
        : jet(directory / "jet.csv"), drops(directory / "drops.csv"), events(directory / "events.csv"),
          neck(directory / "neck.csv") {
        jet << "t_us,L_um";
        for (int cell = 0; cell < cellCount; cell++) jet << ",h" << cell;
        jet << "\n";
        drops << "t_us,z_um,R_um,v_ms\n";
        events << "t_us,type,z_um,R_um,v_ms\n";
        neck << "event,t_us,hmin_um" << endl;
        jet.flush(), drops.flush(), events.flush(); // never leave an empty file (synced folders replace them)
    }

    void writeFrame(double time, JetState jetState, const vector<Drop>& flyingDrops) {
        jet << time * toMicroseconds << "," << jetState.length() * toMicrometres;
        for (int cell = 0; cell < cellCount; cell++) jet << "," << sqrt(jetState.squaredRadius(cell)) * toMicrometres;
        jet << endl;
        for (const Drop& drop : flyingDrops)
            drops << format("{},{},{},{}\n", time * toMicroseconds, drop.position * toMicrometres,
                            drop.radius() * toMicrometres, drop.velocity() * toMetresPerSecond);
        drops.flush(), neck.flush();
    }

    void writeEvent(const string& type, double time, const Drop& drop) {
        events << format("{},{},{},{},{}", time * toMicroseconds, type, drop.position * toMicrometres,
                         drop.radius() * toMicrometres, drop.velocity() * toMetresPerSecond) << endl;
    }

    void writeNeck(int pinchCount, double time, double neckRadius) {
        neck << pinchCount << "," << time * toMicroseconds << "," << neckRadius * toMicrometres << "\n";
    }
};

// The breakup model's inputs as rows of "quantity,value,unit", for plotting scripts.
string runParameterRows() {
    return format("quantity,value,unit\n"
                  "nozzle_radius,{},um\njet_velocity,{},m/s\nforcing_frequency,{},kHz\nnozzle_modulation,{},\n"
                  "liquid_density,{},kg/m^3\nsurface_tension,{},N/m\nviscosity,{},mPa s\nohnesorge,{},\nweber,{},\n"
                  "capillary_time,{},us\ncell_count,{},\npinch_radius,{},nm\n",
                  jetRadius * 1e6, jetVelocity, forcingAngularFrequency / (2 * PI) / capillaryTime * 1e-3,
                  nozzleModulation, liquidDensity, surfaceTension, viscosity * 1e3, ohnesorgeNumber,
                  meanNozzleVelocity * meanNozzleVelocity, toMicroseconds, cellCount, pinchRadius * jetRadius * 1e9);
}

double secondsSince(chrono::steady_clock::time_point start) {
    return chrono::duration<double>(chrono::steady_clock::now() - start).count();
}

// Simulation

int main(int argc, char** argv) {
    cout << unitbuf;
    if (argc > 1) endTime = stod(argv[1]) * 1e-6 / capillaryTime;
    const auto runDirectory = createRunDirectory();
    OutputFiles output(runDirectory);
    ofstream parameterFile(runDirectory / "parameters.csv");
    parameterFile << runParameterRows();

    // ACOUSTIC STANDING WAVE: settings (acousticWave.hpp)
    const AcousticWave acousticWave = makeAcousticWave(demoAcousticWave, liquidDensity, jetRadius, capillaryTime);
    ofstream(runDirectory / "acoustic.txt") << describeAcousticWave(acousticWave);
    parameterFile << acousticWaveParameterRows(acousticWave);
    parameterFile.close();

    cout << format("Oh = {:.4f}, We = {:.3f}, tc = {:.3f} us, f = {:.2f} kHz, N = {}\n", ohnesorgeNumber,
                   meanNozzleVelocity * meanNozzleVelocity, toMicroseconds,
                   forcingAngularFrequency / (2 * PI) / capillaryTime * 1e-3, cellCount)
         << describeAcousticWave(acousticWave) << "Output: " << runDirectory.string() << "\n";

    const auto start = chrono::steady_clock::now();
    auto jetIntegrator = buildJetIntegrator(hemisphericalInitialState(), acousticWave);
    cout << format("Integrator built in {:.1f} s (order {})\n", secondsSince(start), jetIntegrator.get_order());

    vector<Drop> drops;
    int pinchCount = 0, caughtDrop = -1;
    // After every integrator step: move the drops along, log the neck radius, and stop if the tip has caught a drop
    auto afterEachStep = [&](taylor_adaptive<double>& integrator) {
        JetState jet{integrator.get_state_data()};
        advanceDrops(drops, integrator.get_time(), acousticWave);
        if (int neckCell = findNeck(jet, 0.5); neckCell >= 0)
            output.writeNeck(pinchCount, integrator.get_time(), sqrt(jet.squaredRadius(neckCell)));
        if (caughtDrop < 0) caughtDrop = findDropCaughtByTip(jet, drops);
        return caughtDrop < 0;
    };

    int frameCount = 0;
    for (double frameTime = 0.0; frameTime <= endTime;) {
        jetIntegrator.propagate_until(frameTime, kw::callback = afterEachStep, kw::max_delta_t = frameInterval);
        const double t = jetIntegrator.get_time();
        JetState jet{jetIntegrator.get_state_data()};
        advanceDrops(drops, t, acousticWave);
        if (!isPhysical(jet)) {
            cout << format("Solver failed at t = {:.2f} us\n", t * toMicroseconds);
            break;
        }
        if (int neckCell = findNeck(jet, pinchRadius * (1.0 + 1e-6)); neckCell >= 0) {
            drops.push_back(pinchOff(jet, neckCell, t));
            const Drop& drop = drops.back();
            output.writeEvent("pinch", t, drop);
            cout << format("t = {:7.2f} us  pinch-off at z = {:6.1f} um   R = {:5.2f} um  v = {:.2f} m/s\n",
                           t * toMicroseconds, jet.length() * toMicrometres, drop.radius() * toMicrometres,
                           drop.velocity() * toMetresPerSecond);
            pinchCount++;
            continue;
        }
        if (caughtDrop >= 0) {
            absorbDropIntoTip(jet, drops[caughtDrop]);
            output.writeEvent("absorbed", t, drops[caughtDrop]);
            drops.erase(drops.begin() + caughtDrop);
            caughtDrop = -1;
            continue;
        }

        // Output time
        coalesceTouchingDrops(drops);
        output.writeFrame(t, jet, drops);
        frameTime += frameInterval;
        if (++frameCount % 10 == 0) // heartbeat every 10 frames
            cout << format("t = {:7.2f} us  total jet length = {:6.1f} um\n", t * toMicroseconds,
                           jet.length() * toMicrometres);
    }
    cout << format("Done in {:.1f} s, {} pinch-offs\n", secondsSince(start), pinchCount);
}
