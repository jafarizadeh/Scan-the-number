import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createSurveillanceClient } from "../api/createSurveillanceClient";
import type { Reading } from "../types/reading";
import type { RouletteNumber, SurveillanceMode } from "../types/roulette";
import type { SurveillanceSession, SurveillanceStatus } from "../types/session";
import type { SystemStatus } from "../types/system";
import type { HistoryLimit } from "../utils/history";
import { useAlarm } from "./useAlarm";

const MATCH_POPUP_DURATION_MS = 3600;

export function useSurveillanceController() {
  const client = useMemo(() => createSurveillanceClient(), []);
  const liveCleanupRef = useRef<null | (() => void)>(null);
  const matchLockUntilRef = useRef(0);
  const lastPopupReadingIdRef = useRef<string | null>(null);

  const { playAlarm, stopAlarm } = useAlarm();

  const [mode, setMode] = useState<SurveillanceMode>("real");
  const [targetNumber, setTargetNumber] = useState<RouletteNumber>(12);
  const [cameraEnabled, setCameraEnabled] = useState(true);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [armEnabled, setArmEnabled] = useState(true);
  const [status, setStatus] = useState<SurveillanceStatus>("ready");
  const [statusMessage, setStatusMessage] = useState(
    "Prêt. Sélectionnez un numéro puis démarrez.",
  );

  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    camera: "ready",
    backend: "disconnected",
    pico: "unavailable",
    actuator: "unavailable",
    alarm: "ready",
  });

  const [session, setSession] = useState<SurveillanceSession | null>(null);
  const [readings, setReadings] = useState<Reading[]>([]);
  const [latestReading, setLatestReading] = useState<Reading | null>(null);
  const [historyLimit, setHistoryLimit] = useState<HistoryLimit>(100);
  const [matchPopupNumber, setMatchPopupNumber] =
    useState<RouletteNumber | null>(null);

  const stopLiveSubscription = useCallback(() => {
    liveCleanupRef.current?.();
    liveCleanupRef.current = null;
  }, []);

  const showMatch = useCallback(
    (reading: Reading) => {
      const now = Date.now();

      if (now < matchLockUntilRef.current) {
        return;
      }

      matchLockUntilRef.current = now + MATCH_POPUP_DURATION_MS + 600;
      setMatchPopupNumber(reading.number);
      setStatus("match-confirmed");
      setStatusMessage(`Correspondance confirmée : ${reading.number}.`);

      if (soundEnabled) {
        void playAlarm(MATCH_POPUP_DURATION_MS);
      }

      window.setTimeout(() => {
        setMatchPopupNumber(null);
        setStatus((current) =>
          current === "match-confirmed" ? "monitoring" : current,
        );
      }, MATCH_POPUP_DURATION_MS);
    },
    [playAlarm, soundEnabled],
  );

  const acceptReading = useCallback(
    (reading: Reading) => {
      setLatestReading(reading);

      const shouldStoreReading =
        reading.persisted &&
        (reading.source === "real-camera" || reading.source === "backend-live");

      if (shouldStoreReading) {
        setReadings((current) => {
          const exists = current.some((item) => item.id === reading.id);
          if (exists) return current;
          return [reading, ...current].slice(0, 2000);
        });
      }

      if (reading.isMatch && lastPopupReadingIdRef.current !== reading.id) {
        lastPopupReadingIdRef.current = reading.id;
        showMatch(reading);
      }
    },
    [showMatch],
  );

  useEffect(() => {
    client
      .getSystemStatus()
      .then(setSystemStatus)
      .catch(() => {
        setSystemStatus((current) => ({
          ...current,
          backend: "disconnected",
        }));
      });
  }, [client]);

  const startMonitoring = useCallback(async () => {
    stopLiveSubscription();
    lastPopupReadingIdRef.current = null;
    lastPopupReadingIdRef.current = null;

    try {
      const createdSession = await client.startSession({
        mode,
        targetNumber,
        cameraEnabled,
        soundEnabled,
      });

      setSession(createdSession);
      setStatus("monitoring");
      setStatusMessage(
        mode === "real"
          ? "Jeu réel actif : attente des lectures validées par caméra/backend."
          : "Mode test actif : simulation sans sauvegarde.",
      );

      liveCleanupRef.current = client.subscribeLive({
        sessionId: createdSession.id,
        onReading: acceptReading,
        onStatus: (partial) => {
          setSystemStatus((current) => ({
            ...current,
            ...partial,
          }));
        },
        onError: (message) => {
          setStatusMessage(message);
        },
      });
    } catch {
      setStatus("fault");
      setStatusMessage("Impossible de démarrer la surveillance.");
    }
  }, [
    acceptReading,
    cameraEnabled,
    client,
    mode,
    soundEnabled,
    stopLiveSubscription,
    targetNumber,
  ]);

  useEffect(() => {
    if (status !== "monitoring" && status !== "match-confirmed") {
      return;
    }

    const timer = window.setInterval(() => {
      client
        .getBackendStatus()
        .then((backendStatus) => {
          const rawReading = backendStatus.latest_reading;

          if (!rawReading) {
            return;
          }

          const reading: Reading = {
            id: rawReading.id,
            sessionId: rawReading.sessionId ?? rawReading.session_id,
            number: rawReading.number,
            confidence: rawReading.confidence ?? 1,
            capturedAt: rawReading.capturedAt ?? rawReading.captured_at,
            source: rawReading.source ?? "real-camera",
            persisted: rawReading.persisted ?? true,
            isMatch: rawReading.isMatch ?? rawReading.is_match ?? false,
          };

          acceptReading(reading);
        })
        .catch(() => {
          // Keep the UI running even if one polling request fails.
        });
    }, 500);

    return () => window.clearInterval(timer);
  }, [acceptReading, client, status]);

  const stopMonitoring = useCallback(async () => {
    stopLiveSubscription();
    stopAlarm();

    if (!session) {
      setStatus("stopped");
      setStatusMessage("Surveillance arrêtée.");
      return;
    }

    try {
      const stoppedSession = await client.stopSession(session.id);
      setSession(stoppedSession);
      setStatus("stopped");
      setStatusMessage("Surveillance arrêtée.");
    } catch {
      setStatus("fault");
      setStatusMessage("Erreur pendant l'arrêt de la surveillance.");
    }
  }, [client, session, stopAlarm, stopLiveSubscription]);

  const resetHistory = useCallback(async () => {
    await client.clearReadings();
    setReadings([]);
    setLatestReading(null);
    setStatusMessage("Historique local réinitialisé.");
  }, [client]);

  const runSingleTestReading = useCallback(async () => {
    const reading = await client.runTestReading({
      targetNumber,
      forceMatch: false,
    });

    acceptReading(reading);
    setStatusMessage("Lecture test effectuée. Donnée non sauvegardée.");
  }, [acceptReading, client, targetNumber]);

  const runMatchTestReading = useCallback(async () => {
    const reading = await client.runTestReading({
      targetNumber,
      forceMatch: true,
    });

    acceptReading(reading);
    setStatusMessage("Match test effectué. Donnée non sauvegardée.");
  }, [acceptReading, client, targetNumber]);

  const changeTargetNumber = useCallback(
    async (nextTargetNumber: RouletteNumber) => {
      setTargetNumber(nextTargetNumber);

      try {
        await client.setTargetNumber(nextTargetNumber);
        setStatusMessage(`Numéro cible mis à jour : ${nextTargetNumber}.`);
      } catch {
        setStatusMessage("Numéro cible modifié localement, mais non appliqué au backend.");
      }
    },
    [client],
  );

  const changeSoundEnabled = useCallback(
    async (enabled: boolean) => {
      setSoundEnabled(enabled);

      try {
        await fetch(
          `${window.location.protocol}//${window.location.hostname}:8000/api/v1/settings`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              cameraEnabled,
              soundEnabled: enabled,
              armEnabled,
              soundDelaySeconds: 0,
              armDelaySeconds: 0,
              armExtensionPercent: 100,
            }),
          },
        );
      } catch {
        setStatusMessage("Alerte sonore modifiée localement, mais non appliquée au backend.");
      }
    },
    [cameraEnabled, armEnabled],
  );

  const changeArmEnabled = useCallback(
    async (enabled: boolean) => {
      setArmEnabled(enabled);

      try {
        await fetch(
          `${window.location.protocol}//${window.location.hostname}:8000/api/v1/settings`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              cameraEnabled,
              soundEnabled,
              armEnabled: enabled,
              soundDelaySeconds: 0,
              armDelaySeconds: 0,
              armExtensionPercent: 100,
            }),
          },
        );
      } catch {
        setStatusMessage("Bras modifié localement, mais non appliqué au backend.");
      }
    },
    [cameraEnabled, soundEnabled],
  );

  const changeMode = useCallback(
    async (nextMode: SurveillanceMode) => {
      if (status === "monitoring" || status === "match-confirmed") {
        await stopMonitoring();
      }

      setMode(nextMode);
      setStatusMessage(
        nextMode === "real"
          ? "Mode réel : la caméra/backend fournit les lectures."
          : "Mode test : simulation sans sauvegarde.",
      );
    },
    [status, stopMonitoring],
  );

  return {
    mode,
    setMode: changeMode,

    targetNumber,
    setTargetNumber: changeTargetNumber,

    cameraEnabled,
    setCameraEnabled,

    soundEnabled,
    setSoundEnabled: changeSoundEnabled,

    armEnabled,
    setArmEnabled: changeArmEnabled,

    status,
    statusMessage,
    systemStatus,

    session,
    readings,
    latestReading,

    historyLimit,
    setHistoryLimit,

    matchPopupNumber,

    startMonitoring,
    stopMonitoring,
    resetHistory,
    runSingleTestReading,
    runMatchTestReading,
  };
}