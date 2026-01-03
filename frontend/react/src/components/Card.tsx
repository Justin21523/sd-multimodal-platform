import type { PropsWithChildren, ReactNode } from "react";

export function Card(props: PropsWithChildren<{ title?: string; right?: ReactNode; className?: string }>) {
  return (
    <section className={`card ${props.className || ""}`.trim()}>
      {(props.title || props.right) && (
        <header className="card__header">
          <div className="card__title">{props.title}</div>
          <div className="card__right">{props.right}</div>
        </header>
      )}
      <div className="card__body">{props.children}</div>
    </section>
  );
}
