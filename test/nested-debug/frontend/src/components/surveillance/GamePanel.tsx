import type { RouletteNumber, SurveillanceMode } from "../../types/roulette";
import type { SurveillanceStatus } from "../../types/session";

interface GamePanelProps {
  mode: SurveillanceMode;
  status: SurveillanceStatus;
  cameraEnabled: boolean;
  detectedNumber: RouletteNumber | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  streamUrl?: string | null;
  cameraError: string | null;
  onCameraError?: () => void;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
}

export function GamePanel({
  mode,
  status,
  cameraEnabled,
  detectedNumber,
  videoRef,
  streamUrl,
  cameraError,
  onCameraError,
  onStart,
  onStop,
  onReset,
}: GamePanelProps) {
  const running = status === "monitoring" || status === "match-confirmed";
  const showPlaceholder = !cameraEnabled || Boolean(cameraError);

  const cameraStatusLabel = cameraError
    ? cameraError
    : cameraEnabled
      ? "Caméra prête"
      : "Caméra arrêtée";

  return (
    <section className="game-panel" aria-label="Jeu réel">
      <header className="panel-heading horizontal">
        <div>
          <h2>Jeu réel</h2>
          <p>Calage caméra et zone de lecture</p>
        </div>

        <span className={`live-badge ${mode === "real" ? "real" : "test"}`}>
          <span />
          {mode === "real" ? "Live" : "Test"}
        </span>
      </header>

      <div className="camera-stage">
        {streamUrl ? (
          <img
            className="backend-camera-stream"
            src={streamUrl}
            alt="Flux caméra backend"
            onError={onCameraError}
          />
        ) : (
          <video ref={videoRef} autoPlay muted playsInline />
        )}

        <div className={`camera-overlay ${showPlaceholder ? "visible" : ""}`} />

        <div className="roi-frame roi-scan-frame">
          <span className="roi-label">ZONE OCR</span>
          <strong>{detectedNumber ?? "—"}</strong>
        </div>

        <div className="camera-footer">
          <span>Zone numérique</span>
          <span className={cameraEnabled && !cameraError ? "ready" : "stopped"}>
            {cameraStatusLabel}
          </span>
        </div>
      </div>

      <div className="game-actions">
        <button
          type="button"
          className="primary-action"
          onClick={onStart}
          disabled={running}
        >
          {running ? "En cours" : "Démarrer"}
        </button>

        <button type="button" className="secondary-action" onClick={onReset}>
          Reset
        </button>

        <button type="button" className="danger-action compact" onClick={onStop}>
          Stop
        </button>
      </div>
    </section>
  );
}