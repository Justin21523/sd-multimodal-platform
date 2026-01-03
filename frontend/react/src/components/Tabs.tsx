export type TabKey = "generate" | "queue" | "assets" | "system";

export function Tabs(props: { value: TabKey; onChange: (v: TabKey) => void }) {
  const items: Array<{ key: TabKey; label: string }> = [
    { key: "generate", label: "生成" },
    { key: "queue", label: "佇列" },
    { key: "assets", label: "資產" },
    { key: "system", label: "系統" }
  ];

  return (
    <nav className="tabs" aria-label="Navigation Tabs">
      {items.map((t) => (
        <button
          key={t.key}
          type="button"
          className={`tab ${props.value === t.key ? "tab--active" : ""}`}
          onClick={() => props.onChange(t.key)}
        >
          {t.label}
        </button>
      ))}
    </nav>
  );
}
