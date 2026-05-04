import type { ReactNode } from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

type FeatureItem = {
  title: string;
  emoji: string;
  description: ReactNode;
  to: string;
  cta: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: "Teoria, spiegata bene",
    emoji: "📚",
    description: (
      <>
        Sei articoli su classificazione sbilanciata, metriche per fraud
        detection, feature temporali, split temporale, modelli supervisionati,
        threshold tuning.
      </>
    ),
    to: "/docs/category/teoria",
    cta: "Esplora la teoria",
  },
  {
    title: "Pipeline production-ready",
    emoji: "⚙️",
    description: (
      <>
        Codice modulare in Python con scikit-learn + XGBoost, feature
        engineering time-aware, gestione class imbalance, threshold
        optimization, CLI fraud-train.
      </>
    ),
    to: "/docs/scelte-tecniche/architettura",
    cta: "Vedi l'architettura",
  },
  {
    title: "Trade-off documentati",
    emoji: "🎯",
    description: (
      <>
        Ogni scelta è motivata: perché LogReg+RF+XGB, come gestire imbalance,
        come scegliere la soglia operativa, dove sono i limiti del modello.
      </>
    ),
    to: "/docs/scelte-tecniche/scelte-modello",
    cta: "Leggi le decisioni",
  },
];

function Feature({ title, emoji, description, to, cta }: FeatureItem) {
  return (
    <div className={clsx("col col--4")}>
      <div className={styles.featureCard}>
        <div className={styles.featureEmoji} role="img" aria-label={title}>
          {emoji}
        </div>
        <Heading as="h3" className={styles.featureTitle}>
          {title}
        </Heading>
        <p className={styles.featureDesc}>{description}</p>
        <Link
          className={clsx(
            "button button--primary button--sm",
            styles.featureCta,
          )}
          to={to}
        >
          {cta}
        </Link>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="text--center" style={{ marginBottom: "3rem" }}>
          <Heading as="h2">Cosa trovi in questa documentazione</Heading>
          <p
            style={{
              color: "var(--ifm-color-emphasis-700)",
              maxWidth: 700,
              margin: "0 auto",
            }}
          >
            Ogni sezione è autocontenuta. Leggi nell'ordine se vuoi una
            progressione didattica, oppure salta direttamente all'argomento che
            ti serve.
          </p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
