import type { Reading } from "../../types/reading";
import type { RouletteNumber } from "../../types/roulette";
import { formatConfidence, getNumberClass } from "../../utils/roulette";

interface RealTimeComparisonProps {
  targetNumber: RouletteNumber;
  latestReading: Reading | null;
  cameraEnabled: boolean;
  totalReadings: number;
  statusMessage: string;
  onSingleTest: () => void;
  onMatchTest: () => void;
}

export function RealTimeComparison({
  targetNumber,
  latestReading,
  cameraEnabled,
  totalReadings,
  statusMessage,
  onSingleTest,
  onMatchTest,
}: RealTimeComparisonProps) {
  const detectedNumber = latestReading?.number ?? null;
  const isMatch = detectedNumber === targetNumber;

  return (
    <section className={`comparison-card ${isMatch ? "match-state" : ""}`}>
      <header className="panel-heading horizontal">
        <div>
          <h2>Comparaison</h2>
          <p>Cible et lecture validée</p>
        </div>

        <span className={`device-pill ${cameraEnabled ? "online" : "offline"}`}>
          <span />
          Caméra
        </span>
      </header>

      <div className="comparison-main compact">
        <article className="comparison-number-card">
          <span>Cible</span>
          <strong className={getNumberClass(targetNumber)}>{targetNumber}</strong>
        </article>

        <article className="comparison-number-card">
          <span>Lecture</span>
          <strong className={getNumberClass(detectedNumber)}>
            {detectedNumber ?? "—"}
          </strong>
          <small>{formatConfidence(latestReading?.confidence ?? null)}</small>
        </article>
      </div>

      <div className={`comparison-message ${isMatch ? "success-message" : ""}`}>
        {isMatch
          ? "Correspondance confirmée. Action exécutée."
          : statusMessage}
      </div>

      <div className="test-actions">
        <button type="button" className="mini-button" onClick={onSingleTest}>
          Test lecture
        </button>

        <button
          type="button"
          className="mini-button success"
          onClick={onMatchTest}
        >
          Test match
        </button>
      </div>

      <div className="total-readings-line">
  Historique réel : {totalReadings} lecture{totalReadings > 1 ? "s" : ""}
</div>
    </section>
  );
}