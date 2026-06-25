// ============================================================================
// Drone System — SysML v2 Model
// Covers: structural composition, behavioral actions, operational states,
//         ports, connections, attributes, and embedded requirements.
// ============================================================================

package DroneSystem {

    doc /* Complete SysML v2 model of an autonomous multi-rotor drone.
         Structural layer: part/attribute/port definitions.
         Behavioral layer: action and state definitions.
         Embedded requirements: key design constraints. */

    // ─────────────────────────────────────────────────────────────────────────
    // Structural Layer
    // ─────────────────────────────────────────────────────────────────────────

    part def DroneSystem {
        doc /* Top-level autonomous drone system.  Integrates propulsion, power,
             avionics, communication, navigation, and payload subsystems. */
        attribute maxSpeed         : Real;    // 15.0 m/s
        attribute maxAltitude      : Real;    // 120.0 m AGL
        attribute endurance        : Real;    // 30.0 min at max payload
        attribute payloadCapacity  : Real;    // 0.5 kg
        attribute maxAllUpWeight   : Real;    // 2.0 kg
        attribute operatingTempMin : Real;    // -10 °C
        attribute operatingTempMax : Real;    // 45 °C

        part flightControl  : FlightControlUnit;
        part power          : PowerSystem;
        part propulsion     : PropulsionSystem;
        part communication  : CommunicationModule;
        part navigation     : NavigationSystem;
        part payload        : PayloadSystem;
    }

    // ── Flight Control Unit ──────────────────────────────────────────────────

    part def FlightControlUnit {
        doc /* Autopilot unit.  Fuses sensor data, runs cascaded attitude/position
             controllers, manages waypoint mission execution, and triggers failsafe
             responses.  Update rate: attitude 400 Hz, position 50 Hz. */
        part attitudeCtrl  : AttitudeController;
        part waypointNav   : WaypointNavigator;
        part failsafeMgr   : FailsafeManager;
        port rcInput       : RCInputPort;
        port sensorBus     : SensorBusPort;
        port motorBus      : MotorBusPort;
        port mavlinkOut    : MAVLinkOutputPort;
    }

    part def AttitudeController {
        doc /* Cascaded PID attitude stabiliser with feed-forward.
             Inner rate loop bandwidth >= 15 Hz; outer angle loop 5 Hz. */
        attribute rollKp    : Real;    // 6.0
        attribute rollKi    : Real;    // 0.5
        attribute rollKd    : Real;    // 0.2
        attribute pitchKp   : Real;    // 6.0
        attribute pitchKi   : Real;    // 0.5
        attribute pitchKd   : Real;    // 0.2
        attribute yawKp     : Real;    // 3.0
        attribute yawKi     : Real;    // 0.1
        attribute yawKd     : Real;    // 0.1
        attribute loopRateHz: Real;    // 400
    }

    part def WaypointNavigator {
        doc /* GPS-guided waypoint mission executor.
             Supports up to 99 waypoints; cross-track error <= 2 m. */
        attribute maxWaypoints    : Real;   // 99
        attribute crossTrackMaxM  : Real;   // 2.0
        attribute approachRadiusM : Real;   // 1.5
    }

    part def FailsafeManager {
        doc /* Monitors RC link, battery SoC, and geofence boundaries.
             Triggers hover, RTH, or auto-land on threshold violations. */
        attribute linkLossTimeoutSec : Real;   // 2.0
        attribute rthSoCThreshold    : Real;   // 0.20
        attribute autoLandSoCThresh  : Real;   // 0.05
        attribute maxAltitudeLimitM  : Real;   // 120.0
    }

    // ── Power System ─────────────────────────────────────────────────────────

    part def PowerSystem {
        doc /* Battery pack plus power distribution.  Supplies 25.2 V bus to
             propulsion ESCs and regulated 5 V to avionics rail. */
        part battery : BatteryPack;
        part pdb     : PowerDistributionBoard;
        port powerOut: PowerOutputPort;
    }

    part def BatteryPack {
        doc /* LiPo 6S, 5000 mAh, nominal 22.2 V.  BMS prevents overcharge
             (>4.2 V/cell) and over-discharge (<3.0 V/cell).  Min 200 cycles. */
        attribute nominalVoltageV  : Real;   // 22.2
        attribute capacityAh       : Real;   // 5.0
        attribute energyWh         : Real;   // 22.0
        attribute minCycles        : Real;   // 200
        attribute maxCellVoltage   : Real;   // 4.2
        attribute minCellVoltage   : Real;   // 3.0
        attribute stateOfCharge    : Real;   // 0.0–1.0  (runtime value)
        attribute socAccuracyPct   : Real;   // 3.0
    }

    part def PowerDistributionBoard {
        doc /* Routes 25.2 V to four ESC rails; provides 5 V regulated avionics bus. */
        attribute avionicsVoltV  : Real;   // 5.0
        attribute maxCurrentA    : Real;   // 80.0
    }

    // ── Propulsion System ────────────────────────────────────────────────────

    part def PropulsionSystem {
        doc /* Quadrotor propulsion: 4 × (motor + propeller + ESC) in X-configuration.
             Minimum thrust-to-weight ratio 2:1 at max all-up weight. */
        part motorFL : BrushlessMotor;
        part motorFR : BrushlessMotor;
        part motorRL : BrushlessMotor;
        part motorRR : BrushlessMotor;
        part propFL  : Propeller;
        part propFR  : Propeller;
        part propRL  : Propeller;
        part propRR  : Propeller;
        part escFL   : ESC;
        part escFR   : ESC;
        part escRL   : ESC;
        part escRR   : ESC;
    }

    part def BrushlessMotor {
        doc /* KV920 brushless DC motor.  Static thrust >= 500 g at 100% throttle.
             Operating temperature 0–60 °C. */
        attribute kv          : Real;   // 920 RPM/V
        attribute maxThrustG  : Real;   // 714
        attribute maxTempC    : Real;   // 60
    }

    part def Propeller {
        doc /* 10-inch / 4.5-pitch carbon-fibre two-blade propeller. */
        attribute diameterIn : Real;   // 10
        attribute pitchIn    : Real;   // 4.5
    }

    part def ESC {
        doc /* 40 A DSHOT-compatible electronic speed controller.
             Throttle command response time <= 5 ms. */
        attribute maxCurrentA    : Real;   // 40
        attribute responseTimeMs : Real;   // 5
        attribute protocolType   : Real;   // DSHOT600
    }

    // ── Communication Module ─────────────────────────────────────────────────

    part def CommunicationModule {
        doc /* Handles FHSS RC uplink (SBUS) and MAVLink 2.0 telemetry downlink.
             RC round-trip latency <= 20 ms; telemetry range >= 1000 m. */
        part rcReceiver       : RCReceiver;
        part telemetryXceiver : TelemetryTransceiver;
        port telemetryOut     : TelemetryOutputPort;
        port rcOut            : RCOutputPort;
    }

    part def RCReceiver {
        doc /* 2.4 GHz FHSS RC receiver.  Outputs SBUS at 100 Hz.  16 channels. */
        attribute frequencyMHz  : Real;   // 2400
        attribute channels      : Real;   // 16
        attribute outputRateHz  : Real;   // 100
        attribute latencyMs     : Real;   // 20
    }

    part def TelemetryTransceiver {
        doc /* 915 MHz half-duplex transceiver.  MAVLink 2.0.  AES-128 encrypted
             command channel.  Range >= 1000 m.  Data rate 57.6 kbps. */
        attribute frequencyMHz  : Real;   // 915
        attribute rangeM        : Real;   // 1000
        attribute dataRateKbps  : Real;   // 57.6
        attribute linkMarginDb  : Real;   // 6.0  minimum at 1000 m
    }

    // ── Navigation System ────────────────────────────────────────────────────

    part def NavigationSystem {
        doc /* Integrated position/orientation solution.  Fuses GNSS, IMU, and
             barometer via EKF.  Provides obstacle detection and geofencing. */
        part gnss            : GNSSReceiver;
        part imu             : ImuSensor;
        part barometer       : Barometer;
        part obstacleDetect  : ObstacleDetector;
        port positionOut     : PositionOutputPort;
    }

    part def GNSSReceiver {
        doc /* Dual-band GPS+GLONASS receiver.  CEP <= 1.5 m.  Cold fix <= 30 s. */
        attribute cepM           : Real;   // 1.5
        attribute coldFixSec     : Real;   // 30
        attribute warmFixSec     : Real;   // 5
        attribute updateRateHz   : Real;   // 10
    }

    part def ImuSensor {
        doc /* 6-axis MEMS IMU (3-axis accelerometer + 3-axis gyroscope).
             Sample rate >= 400 Hz; internal low-pass filter configurable. */
        attribute sampleRateHz  : Real;   // 1000
        attribute gyroRangedps  : Real;   // 2000
        attribute accelRangeG   : Real;   // 16
    }

    part def Barometer {
        doc /* Barometric pressure sensor.  Altitude accuracy ±0.5 m static. */
        attribute altAccuracyM  : Real;   // 0.5
        attribute sampleRateHz  : Real;   // 50
    }

    part def ObstacleDetector {
        doc /* ToF / ultrasonic obstacle detection.  Detection range >= 5 m.
             Outputs an obstacle map to the WaypointNavigator. */
        attribute detectionRangeM : Real;   // 5.0
        attribute updateRateHz    : Real;   // 20
    }

    // ── Payload System ───────────────────────────────────────────────────────

    part def PayloadSystem {
        doc /* Stabilised camera payload.  Accepts up to 500 g total mass.
             Provides 12 V / 2 A regulated power to the payload. */
        part camera : Camera;
        part gimbal : Gimbal;
        port payloadPower   : PayloadPowerPort;
        port videoOut       : VideoOutputPort;
    }

    part def Camera {
        doc /* 4K / 30 fps CMOS camera.  Field of view 84° horizontal. */
        attribute resolutionK : Real;   // 4
        attribute fps         : Real;   // 30
        attribute fovDeg      : Real;   // 84
    }

    part def Gimbal {
        doc /* 3-axis brushless stabilised gimbal.  Stabilisation accuracy ±0.01°.
             Tilt range -90° to 0° (nadir to forward). */
        attribute stabilAccuracyDeg : Real;   // 0.01
        attribute tiltRangeDeg      : Real;   // 90
        attribute rollRangeDeg      : Real;   // 45
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Port Definitions
    // ─────────────────────────────────────────────────────────────────────────

    port def RCInputPort         { doc /* SBUS RC uplink from RC receiver to FCU. */ }
    port def SensorBusPort       { doc /* Internal sensor data bus: IMU, baro, GNSS → FCU. */ }
    port def MotorBusPort        { doc /* DSHOT motor command bus: FCU → ESCs. */ }
    port def MAVLinkOutputPort   { doc /* MAVLink 2.0 stream from FCU to telemetry transceiver. */ }
    port def PowerOutputPort     { doc /* 25.2 V battery bus output to ESCs. */ }
    port def TelemetryOutputPort { doc /* RF telemetry downlink to ground station. */ }
    port def RCOutputPort        { doc /* Decoded RC commands to FCU. */ }
    port def PositionOutputPort  { doc /* EKF position / velocity / attitude solution. */ }
    port def PayloadPowerPort    { doc /* Regulated 12 V / 2 A payload power feed. */ }
    port def VideoOutputPort     { doc /* Digital video stream to ground station. */ }

    // ─────────────────────────────────────────────────────────────────────────
    // Connection Definitions
    // ─────────────────────────────────────────────────────────────────────────

    connection def PowerFlow       { doc /* DC electrical power path. */ }
    connection def ControlDataFlow { doc /* Digital control command path. */ }
    connection def SensorDataFlow  { doc /* Digital sensor measurement path. */ }
    connection def RFLink          { doc /* Radio frequency over-the-air link. */ }
    connection def MechanicalLink  { doc /* Rigid mechanical mounting interface. */ }
    connection def VideoDataFlow   { doc /* Digital video data path. */ }

    // ─────────────────────────────────────────────────────────────────────────
    // Behavioral Layer — Actions
    // ─────────────────────────────────────────────────────────────────────────

    action def PerformMission {
        doc /* Top-level mission action.  Sequences: preflight → takeoff →
             [waypoint loop] → return-to-home → land.  Returns mission result. */
        in  missionPlan   : Real;
        out missionResult : Real;
    }

    action def Preflight {
        doc /* Pre-arming check sequence: sensor self-test, GNSS fix acquisition,
             ESC calibration check, battery SoC check (>= 30%), and geofence load. */
        out preflightOk : Real;
    }

    action def ArmMotors {
        doc /* Arms the ESC/motor subsystem after verifying throttle at minimum
             position and all preflight checks passed. */
        in  preflightOk : Real;
    }

    action def TakeOff {
        doc /* Autonomously lifts drone from ground to target hover altitude.
             Requires GPS fix, SoC >= 30%, and motors armed. */
        in  targetAltitudeM  : Real;
        out reachedAltitude  : Real;
    }

    action def HoverInPlace {
        doc /* GPS position-hold at current altitude.  Maximum drift <= 1.5 m CEP.
             Remains in this state until next waypoint command or operator override. */
        in  holdDurationSec : Real;
    }

    action def NavigateToWaypoint {
        doc /* Flies to a target GPS waypoint at specified altitude and cruise speed.
             Executes obstacle avoidance if obstacleMap is non-empty. */
        in  waypointLat  : Real;
        in  waypointLon  : Real;
        in  waypointAltM : Real;
        in  speedMs      : Real;
        out arrivalFlag  : Real;
    }

    action def ExecuteMissionTask {
        doc /* Carries out the task defined for the current waypoint (e.g., capture
             image, loiter for N seconds, trigger payload action). */
        in  taskCode   : Real;
        out taskResult : Real;
    }

    action def ReturnToHome {
        doc /* Flies drone back to stored home position at RTH altitude (30 m AGL),
             then descends for precision landing. */
        in  homeLat : Real;
        in  homeLon : Real;
    }

    action def Land {
        doc /* Controlled autonomous descent to ground at current position.
             Detects ground contact via barometer/IMU; cuts motors on touch-down. */
        out touchdownFlag : Real;
    }

    action def EmergencyLanding {
        doc /* Immediate controlled descent to ground triggered by critical battery
             (SoC < 5%) or hardware fault.  Overrides any current mission phase. */
        in  faultCode : Real;
    }

    action def DisarmMotors {
        doc /* Disarms ESC/motor subsystem after landing detected or on pilot command. */
    }

    action def RunAttitudeController {
        doc /* Executes cascaded PID attitude loop at 400 Hz.
             Computes per-motor thrust commands from attitude error and setpoint. */
        in  attitudeSetpoint : Real;
        in  imuReading       : Real;
        out motorThrottle    : Real;
    }

    action def RunPositionController {
        doc /* Computes attitude setpoint from position/velocity error at 50 Hz.
             Feeds into RunAttitudeController. */
        in  positionSetpoint : Real;
        in  gnssPosition     : Real;
        out attSetpoint      : Real;
    }

    action def BroadcastTelemetry {
        doc /* Transmits 10 Hz MAVLink 2.0 telemetry: GLOBAL_POSITION_INT,
             ATTITUDE, SYS_STATUS, BATTERY_STATUS, HEARTBEAT. */
        out telemetryFrame : Real;
    }

    action def ProcessRCInput {
        doc /* Decodes SBUS RC frame; maps channel values to attitude setpoints
             (manual mode) or velocity setpoints (position-hold mode). */
        in  sbusFrame  : Real;
        out setpoint   : Real;
    }

    action def MonitorBattery {
        doc /* Samples cell voltages at 1 Hz; runs Coulomb counting + voltage model
             for SoC estimate.  Publishes failsafe events on threshold crossings. */
        in  cellVoltages   : Real;
        out socEstimate    : Real;
        out failsafeEvent  : Real;
    }

    action def DetectObstacles {
        doc /* Processes ToF sensor data at 20 Hz; clusters returns into obstacle
             zones; publishes obstacle map to WaypointNavigator. */
        in  toFData      : Real;
        out obstacleMap  : Real;
    }

    action def EstimateState {
        doc /* Extended Kalman Filter fusing IMU (400 Hz), GNSS (10 Hz), and
             barometer (50 Hz) into position/velocity/attitude estimate. */
        in  imuData      : Real;
        in  gnssData     : Real;
        in  baroData     : Real;
        out stateVector  : Real;
    }

    action def EnforceGeofence {
        doc /* Compares current position against geofence polygons; commands RTH
             or hover if boundary is approached or breached. */
        in  position    : Real;
        in  geofenceDef : Real;
        out violation   : Real;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Behavioral Layer — States
    // ─────────────────────────────────────────────────────────────────────────

    state def DroneOperationalState {
        doc /* Top-level operational state machine for the drone system.
             Transitions are guarded by hardware/software conditions. */

        state Grounded {
            doc /* Drone is on the ground with motors disarmed.  Initial state
                 after power-on or completion of a flight. */
        }

        state PreFlight {
            doc /* Running preflight checks: sensor self-test, GPS fix acquisition,
                 ESC calibration, battery SoC >= 30%.  Transitions to TakingOff
                 on successful arm command. */
        }

        state TakingOff {
            doc /* Climbing from ground to target hover altitude.
                 Transitions to Hovering when target altitude is reached
                 within +/- 0.5 m. */
        }

        state Hovering {
            doc /* Stationary GPS position hold at altitude.  Default stable state
                 between waypoints and during mission pauses.  Transitions to
                 Navigating on next waypoint command. */
        }

        state Navigating {
            doc /* Actively flying towards a target waypoint.  Transitions to
                 Hovering on waypoint arrival, or to ReturnHome on RTH trigger. */
        }

        state ReturnHome {
            doc /* Executing return-to-home manoeuvre: climbs to RTH altitude,
                 flies to home position, transitions to Landing. */
        }

        state Landing {
            doc /* Controlled descent from current position to ground.  Transitions
                 to Grounded on touch-down detection. */
        }

        state Emergency {
            doc /* Critical fault detected (hardware failure, critical battery).
                 Immediately transitions to EmergencyLanding action. */
        }
    }

    state def MotorState {
        doc /* ESC/motor armed/disarmed state. */
        state Disarmed { doc /* Motors safe; no thrust produced. */ }
        state Armed    { doc /* Motors ready; throttle commands accepted. */ }
        state Running  { doc /* Motors spinning; drone in flight. */ }
        state FaultStop { doc /* Motor fault detected; motor stopped. */ }
    }

    state def BatteryState {
        doc /* Battery operational state derived from SoC. */
        state Normal    { doc /* SoC > 20%.  Normal operation. */ }
        state LowBattery { doc /* SoC 15–20%.  RTH warning triggered. */ }
        state Critical  { doc /* SoC 5–15%.  RTH commanded. */ }
        state Emergency { doc /* SoC < 5%.  Auto-land commanded. */ }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Embedded Requirement Definitions
    // ─────────────────────────────────────────────────────────────────────────

    requirement SafetyAltitudeReq {
        doc /* The drone system shall not exceed 120 m altitude above ground level
             under autonomous control in standard operating mode. */
        subject droneSystem;
    }

    requirement EnduranceReq {
        doc /* The drone system shall achieve a minimum of 30 minutes of continuous
             hover flight when carrying the maximum rated payload of 500 g. */
        subject droneSystem;
    }

    requirement LinkBudgetReq {
        doc /* The communication system shall maintain a minimum link budget margin
             of 6 dB at 1000 m slant range under standard atmospheric conditions. */
        subject communicationModule;
    }

    requirement PositionAccuracyReq {
        doc /* The navigation system shall provide position estimates with a
             CEP of 1.5 m or better under open-sky dual-band GNSS conditions. */
        subject navigationSystem;
    }

    requirement FailsafeReq {
        doc /* The flight control unit shall initiate return-to-home within 2 seconds
             of detecting a continuous RC link loss exceeding the timeout threshold. */
        subject flightControlUnit;
    }

    requirement ThrustMarginReq {
        doc /* The propulsion system shall deliver a thrust-to-weight ratio of at
             least 2.0 at maximum all-up weight of 2.0 kg at ISA sea level. */
        subject propulsionSystem;
    }

    requirement PayloadInterfaceReq {
        doc /* The payload interface shall accept any payload up to 500 g and supply
             12 V / 2 A regulated power without interruption during flight. */
        subject payloadSystem;
    }
}
