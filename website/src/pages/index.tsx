import type { ReactNode } from "react";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";

import HomepageFeatures from "@site/src/components/HomepageFeatures";

const KPIs = [
  { label: "PR-AUC atteso", value: "0.85+", hint: "XGBoost full tuning" },
  { label: "Recall@P=0.5", value: "~0.78", hint: "soglia operativa" },
  { label: "Class imbalance", value: "0.5%", hint: "frodi nel dataset" },
];

function HomepageHero(): ReactNode {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className="hero-pw">
      <div className="hero-pw__inner">
        <h1 className="hero-pw__title">{siteConfig.title}</h1>
        <p className="hero-pw__subtitle">{siteConfig.tagline}</p>
        <div className="hero-pw__buttons">
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro"
          >
            Inizia dalla documentazione →
          </Link>
          <Link
            className="button button--outline button--lg"
            style={{ color: "#fff", borderColor: "rgba(255,255,255,0.6)" }}
            to="https://github.com/fedcal/Credit-Card-Transactions-Fraud-Detection-Dataset"
          >
            Codice su GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}

function HomepageKPIs(): ReactNode {
  return (
    <section className="kpi-section">
      <div className="container">
        <div className="text--center" style={{ marginBottom: "2rem" }}>
          <h2 style={{ marginBottom: "0.5rem" }}>Risultati di riferimento</h2>
          <p
            style={{
              color: "var(--ifm-color-emphasis-700)",
              maxWidth: 600,
              margin: "0 auto",
            }}
          >
            Metriche misurate sul holdout test set dopo tuning K-fold del
            modello vincente.
          </p>
        </div>
        <div className="kpi-grid">
          {KPIs.map((kpi) => (
            <div key={kpi.label} className="kpi-card">
              <div className="kpi-card__label">{kpi.label}</div>
              <div className="kpi-card__value">{kpi.value}</div>
              <div className="kpi-card__hint">{kpi.hint}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout title={`${siteConfig.title}`} description={siteConfig.tagline}>
      <HomepageHero />
      <HomepageKPIs />
      <main className="features-section">
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
