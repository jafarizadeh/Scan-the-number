import { useEffect, useState } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  `${window.location.protocol}//${window.location.hostname}:8000`;

interface RobotSettings {
  cameraEnabled: boolean;
  soundEnabled: boolean;
  armEnabled: boolean;
  soundDelaySeconds: number;
  armDelaySeconds: number;
}

const DEFAULT_SETTINGS: RobotSettings = {
  cameraEnabled: true,
  soundEnabled: true,
  armEnabled: true,
  soundDelaySeconds: 0,
  armDelaySeconds: 0,
};

const DELAY_OPTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

export function SettingsPage() {
  const [settings, setSettings] = useState<RobotSettings>(DEFAULT_SETTINGS);
  const [message, setMessage] = useState("Chargement des réglages...");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/v1/settings`)
      .then((response) => response.json())
      .then((data) => {
        setSettings({
          ...DEFAULT_SETTINGS,
          ...data,
          soundDelaySeconds: Math.max(0, Number(data.soundDelaySeconds ?? 0)),
          armDelaySeconds: Math.max(0, Number(data.armDelaySeconds ?? 0)),
        });
        setMessage("Réglages chargés.");
      })
      .catch(() => setMessage("Impossible de charger les réglages backend."));
  }, []);

  function update<K extends keyof RobotSettings>(
    key: K,
    value: RobotSettings[K],
  ) {
    setSettings((current) => ({ ...current, [key]: value }));
    setMessage("Réglages modifiés. Cliquez sur Save / Apply pour appliquer.");
  }

  async function saveSettings() {
    setSaving(true);
    setMessage("Sauvegarde des réglages...");

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      });

      if (!response.ok) {
        throw new Error("Settings request failed");
      }

      const savedSettings = await response.json();
      setSettings({
        ...DEFAULT_SETTINGS,
        ...savedSettings,
        soundDelaySeconds: Math.max(0, Number(savedSettings.soundDelaySeconds ?? 0)),
        armDelaySeconds: Math.max(0, Number(savedSettings.armDelaySeconds ?? 0)),
      });
      setMessage("Réglages sauvegardés et appliqués.");
    } catch {
      setMessage("Erreur pendant la sauvegarde des réglages.");
    } finally {
      setSaving(false);
    }
  }

  async function testPico() {
    setMessage("Test Pico en cours...");

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/pico/test`, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error("Pico test failed");
      }

      setMessage("Commande test envoyée au Pico.");
    } catch {
      setMessage("Impossible d’envoyer la commande test au Pico.");
    }
  }

  return (
    <section className="settings-page">
      <header className="settings-page-header">
        <div>
          <h1>Settings</h1>
          <p>Réglages caméra, alarme sonore et bras robotique.</p>
        </div>
      </header>

      <div className="settings-cards">
        <section className="settings-card">
          <h2>Activation</h2>

          <label className="settings-toggle">
            <span>Caméra</span>
            <input
              type="checkbox"
              checked={settings.cameraEnabled}
              onChange={(event) => update("cameraEnabled", event.target.checked)}
            />
          </label>

          <label className="settings-toggle">
            <span>Bras robotique</span>
            <input
              type="checkbox"
              checked={settings.armEnabled}
              onChange={(event) => update("armEnabled", event.target.checked)}
            />
          </label>
        </section>

        <section className="settings-card">
          <h2>Délais de réaction</h2>

          <label className="settings-field">
            <span>Délai son / secondes</span>
            <select
              value={settings.soundDelaySeconds}
              onChange={(event) =>
                update("soundDelaySeconds", Number(event.target.value))
              }
            >
              {DELAY_OPTIONS.map((value) => (
                <option key={value} value={value}>
                  {value} s
                </option>
              ))}
            </select>
          </label>

          <label className="settings-field">
            <span>Délai bras / secondes</span>
            <select
              value={settings.armDelaySeconds}
              onChange={(event) =>
                update("armDelaySeconds", Number(event.target.value))
              }
            >
              {DELAY_OPTIONS.map((value) => (
                <option key={value} value={value}>
                  {value} s
                </option>
              ))}
            </select>
          </label>
        </section>


      </div>

      <div className="settings-actions">
        <button
          type="button"
          className="settings-save-button"
          onClick={saveSettings}
          disabled={saving}
        >
          {saving ? "Saving..." : "Save / Apply"}
        </button>
      </div>

      <p className="settings-message">{message}</p>
    </section>
  );
}
