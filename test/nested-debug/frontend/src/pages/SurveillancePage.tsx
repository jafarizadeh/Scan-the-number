import { SettingsPanel } from "../components/surveillance/SettingsPanel";
import { GamePanel } from "../components/surveillance/GamePanel";
import { RealTimeComparison } from "../components/surveillance/RealTimeComparison";
import { ReadingsHistory } from "../components/surveillance/ReadingsHistory";
import { MatchPopup } from "../components/surveillance/MatchPopup";
import { useCameraPreview } from "../hooks/useCameraPreview";
import { useSurveillanceController } from "../hooks/useSurveillanceController";

export function SurveillancePage() {
  const controller = useSurveillanceController();

  const { videoRef, streamUrl, error: cameraError, setError: setCameraError } = useCameraPreview(
    controller.cameraEnabled,
  );

  return (
    <section className="surveillance-grid" aria-label="Page principale">
      <SettingsPanel
        mode={controller.mode}
        targetNumber={controller.targetNumber}
        cameraEnabled={controller.cameraEnabled}
        soundEnabled={controller.soundEnabled}
        armEnabled={controller.armEnabled}
        onModeChange={controller.setMode}
        onTargetNumberChange={controller.setTargetNumber}
        onCameraToggle={() =>
          controller.setCameraEnabled(!controller.cameraEnabled)
        }
        onSoundToggle={() =>
          controller.setSoundEnabled(!controller.soundEnabled)
        }
        onArmToggle={() =>
          controller.setArmEnabled(!controller.armEnabled)
        }
      />

      <GamePanel
        mode={controller.mode}
        status={controller.status}
        cameraEnabled={controller.cameraEnabled}
        detectedNumber={controller.latestReading?.number ?? null}
        videoRef={videoRef}
        streamUrl={streamUrl}
        cameraError={cameraError}
        onCameraError={() => setCameraError("Flux caméra backend indisponible")}
        onStart={controller.startMonitoring}
        onStop={controller.stopMonitoring}
        onReset={controller.resetHistory}
      />

      <RealTimeComparison
        targetNumber={controller.targetNumber}
        latestReading={controller.latestReading}
        cameraEnabled={controller.cameraEnabled}
        totalReadings={controller.readings.length}
        statusMessage={controller.statusMessage}
        onSingleTest={controller.runSingleTestReading}
        onMatchTest={controller.runMatchTestReading}
      />

      <ReadingsHistory
        readings={controller.readings}
        limit={controller.historyLimit}
        onLimitChange={controller.setHistoryLimit}
      />

      <MatchPopup number={controller.matchPopupNumber} />
    </section>
  );
}