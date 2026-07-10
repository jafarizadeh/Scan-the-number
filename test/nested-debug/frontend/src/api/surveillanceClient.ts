import type { Reading } from "../types/reading";
import type { RouletteNumber, SurveillanceMode } from "../types/roulette";
import type { SurveillanceSession } from "../types/session";
import type { SystemStatus } from "../types/system";

export interface StartSessionInput {
  mode: SurveillanceMode;
  targetNumber: RouletteNumber;
  cameraEnabled: boolean;
  soundEnabled: boolean;
}

export interface SurveillanceClient {
  getSystemStatus(): Promise<SystemStatus>;

  getBackendStatus(): Promise<any>;

  setTargetNumber(targetNumber: number): Promise<void>;

  startSession(input: StartSessionInput): Promise<SurveillanceSession>;

  stopSession(sessionId: string): Promise<SurveillanceSession>;

  getReadings(params: {
    sessionId?: string;
    limit: number;
  }): Promise<Reading[]>;

  clearReadings(): Promise<void>;

  runTestReading(params: {
    targetNumber: RouletteNumber;
    forceMatch: boolean;
  }): Promise<Reading>;

  subscribeLive(params: {
    sessionId: string;
    onReading: (reading: Reading) => void;
    onStatus?: (status: Partial<SystemStatus>) => void;
    onError?: (message: string) => void;
  }): () => void;
}