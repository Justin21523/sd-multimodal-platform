export function StatusPill(props: { label: string; tone?: "info" | "ok" | "warn" | "bad" }) {
  const tone = props.tone ?? "info";
  return <span className={`pill pill--${tone}`}>{props.label}</span>;
}

