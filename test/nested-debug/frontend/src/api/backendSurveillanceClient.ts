import type { Reading } from "../types/reading";
import type { SurveillanceSession } from "../types/session";
import type { SystemStatus } from "../types/system";
import type { StartSessionInput, SurveillanceClient } from "./surveillanceClient";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  `${window.location.protocol}//${window.location.hostname}:8000`;

async function requestJson<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
  });

  if (!response.ok) {
    throw new Error(`Backend request failed: ${response.status} ${path}`);
  }

  return response.json() as Promise<T>;
}

export function createBackendSurveillanceClient(): SurveillanceClient {
  return {
    getSystemStatus() {
      return requestJson<SystemStatus>("/api/v1/system/status");
    },

    getBackendStatus() {
      return requestJson<any>("/api/status");
    },

    async setTargetNumber(targetNumber: number) {
      await requestJson("/api/target", {
        method: "POST",
        body: JSON.stringify({ target_number: targetNumber }),
      });
    },

    startSession(input: StartSessionInput) {
      return requestJson<SurveillanceSession>("/api/v1/sessions", {
        method: "POST",
        body: JSON.stringify(input),
      });
    },

    stopSession(sessionId: string) {
      return requestJson<SurveillanceSession>(
        `/api/v1/sessions/${sessionId}/stop`,
        {
          method: "POST",
        },
      );
    },

    getReadings({ sessionId, limit }) {
      const search = new URLSearchParams();
      search.set("limit", String(limit));

      if (sessionId) {
        search.set("sessionId", sessionId);
      }

      return requestJson<Reading[]>(`/api/v1/readings?${search.toString()}`);
    },

    clearReadings() {
      return requestJson<void>("/api/v1/admin/readings/clear", {
        method: "POST",
      });
    },

    runTestReading({ targetNumber, forceMatch }) {
      return requestJson<Reading>("/api/v1/admin/test-reading", {
        method: "POST",
        body: JSON.stringify({
          targetNumber,
          forceMatch,
        }),
      });
    },

    subscribeLive({ sessionId, onReading, onStatus, onError }) {
      const wsProtocol = API_BASE_URL.startsWith("https") ? "wss" : "ws";
      const url = new URL(API_BASE_URL);
      const socket = new WebSocket(
        `${wsProtocol}://${url.host}/api/v1/live?sessionId=${encodeURIComponent(sessionId)}`,
      );

      socket.addEventListener("message", (event) => {
        try {
          const message = JSON.parse(event.data) as {
            type?: string;
            payload?: unknown;
          };

          if (message.type === "reading.validated") {
            onReading(message.payload as Reading);
          }

          if (message.type === "system.status.changed") {
            onStatus?.(message.payload as Partial<SystemStatus>);
          }
        } catch {
          onError?.("Invalid live message received from backend");
        }
      });

      socket.addEventListener("error", () => {
        onError?.("Live backend WebSocket error");
      });

      return () => {
        socket.close();
      };
    },
  };
}
