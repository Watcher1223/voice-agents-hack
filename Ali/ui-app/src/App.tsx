import { useEffect, useState } from "react";

const FEATURES = [
  {
    icon: "🎧",
    title: "Listens in the background",
    body: "Ali joins your meetings silently. No bot in the call, no awkward join link — just a calm presence on your laptop that follows the conversation.",
  },
  {
    icon: "⚡",
    title: "Acts while you talk",
    body: "Flight searches, calendar holds, doc lookups, drafts. Ali fires off the work in parallel as it hears the ask, so the answers are ready when the meeting ends.",
  },
  {
    icon: "✅",
    title: "Confirms before sending",
    body: "Nothing gets sent or booked without your nod. At the end of the meeting Ali surfaces a single review screen — approve, edit, or discard.",
  },
  {
    icon: "👀",
    title: "Sees what you see",
    body: "PDFs, code, browser tabs, slides. Ali reads the screen alongside the audio so it can resolve “this thing here” without you re-explaining.",
  },
  {
    icon: "🔌",
    title: "Connects to your stack",
    body: "Calendar, mail, search, flights, docs. Plug Ali into the tools you already pay for — it uses them on your behalf.",
  },
  {
    icon: "🔒",
    title: "Private by default",
    body: "Audio and screen context stay on your machine until an action needs the network. You see every outbound call before it goes.",
  },
];

const STEPS = [
  {
    n: "01",
    title: "Open Ali before the call",
    body: "One keystroke. Ali sits in the corner and starts listening. No setup per meeting.",
  },
  {
    n: "02",
    title: "Run the meeting like normal",
    body: "Talk to your team. As asks come up — book the trip, send the brief, find the doc — Ali quietly starts the work in parallel.",
  },
  {
    n: "03",
    title: "Review at the end",
    body: "When the meeting ends, Ali shows you everything it queued up. One click to approve, edit, or skip each item.",
  },
];

const USE_CASES = [
  {
    tag: "Founders",
    title: "Investor calls without the catch-up tax",
    body: "Decks pulled, intros drafted, follow-ups queued. Walk out of the call with the homework already done.",
  },
  {
    tag: "Sales",
    title: "Discovery → next-step in one motion",
    body: "Notes, CRM updates, proposal drafts and the calendar invite — all staged before you’ve closed the laptop.",
  },
  {
    tag: "Operators",
    title: "Standups that actually move tickets",
    body: "Ali turns the conversation into a clean action list, files it where your team already works, and pings the right owners.",
  },
];

function NavBar() {
  return (
    <nav className="nav">
      <a className="nav-brand" href="#top">
        <span className="logo-mark" aria-hidden>
          <svg viewBox="0 0 24 24" width="22" height="22">
            <circle cx="12" cy="12" r="10" fill="none" stroke="#a855f7" strokeWidth="1.6" />
            <circle cx="12" cy="12" r="3.2" fill="#a855f7" />
          </svg>
        </span>
        <span className="logo-text">Ali</span>
      </a>
      <div className="nav-links">
        <a href="#features">Features</a>
        <a href="#how">How it works</a>
        <a href="#use-cases">Use cases</a>
        <a href="#faq">FAQ</a>
      </div>
      <div className="nav-cta">
        <a className="btn btn--ghost" href="#waitlist">Sign in</a>
        <a className="btn btn--primary" href="#waitlist">Get early access</a>
      </div>
    </nav>
  );
}

function Hero() {
  return (
    <header className="hero" id="top">
      <div className="hero-glow" aria-hidden />
      <div className="hero-inner">
        <span className="eyebrow">
          <span className="dot" /> Now in private beta
        </span>
        <h1 className="hero-title">
          The AI that <span className="grad">listens to your meetings</span>
          <br /> and finishes the work for you.
        </h1>
        <p className="hero-sub">
          Ali sits quietly on your laptop, hears what needs to happen, and gets it done in
          parallel — flights booked, calendar held, docs drafted — before you’ve even said goodbye.
        </p>
        <div className="hero-cta">
          <a className="btn btn--primary btn--lg" href="#waitlist">
            Join the waitlist
          </a>
          <a className="btn btn--ghost btn--lg" href="#how">
            See how it works →
          </a>
        </div>
        <div className="hero-trust">
          <span>Built for founders, operators, and teams who run meetings back-to-back.</span>
        </div>
      </div>

      <div className="hero-card" aria-hidden>
        <div className="hero-card-head">
          <span className="hero-card-dot dot--live" />
          <span>Live · Board sync</span>
          <span className="hero-card-time">38:12</span>
        </div>
        <ul className="hero-card-list">
          <li>
            <span className="hcl-tag tag--done">Done</span>
            Found 4 SFO → JFK flights for May 2 (Mon)
          </li>
          <li>
            <span className="hcl-tag tag--queue">Queued</span>
            Hold 30 min Tues w/ Maya — pending your review
          </li>
          <li>
            <span className="hcl-tag tag--draft">Drafting</span>
            Follow-up email to investors with deck v3
          </li>
          <li>
            <span className="hcl-tag tag--hear">Listening</span>
            “Let’s circle back on the pricing page…”
          </li>
        </ul>
      </div>
    </header>
  );
}

function Features() {
  return (
    <section className="section" id="features">
      <div className="section-head">
        <span className="kicker">What Ali does</span>
        <h2>An assistant that actually finishes things.</h2>
        <p>
          Most AI tools take notes. Ali takes <em>action</em> — quietly, in the background,
          and only ships work after you sign off.
        </p>
      </div>
      <div className="grid grid--3">
        {FEATURES.map((f) => (
          <article key={f.title} className="card">
            <div className="card-icon" aria-hidden>{f.icon}</div>
            <h3>{f.title}</h3>
            <p>{f.body}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function HowItWorks() {
  return (
    <section className="section section--alt" id="how">
      <div className="section-head">
        <span className="kicker">How it works</span>
        <h2>Three steps. Zero new habits.</h2>
        <p>You don’t learn a new tool. You run your meeting; Ali runs the chores.</p>
      </div>
      <ol className="steps">
        {STEPS.map((s) => (
          <li key={s.n} className="step">
            <span className="step-n">{s.n}</span>
            <h3>{s.title}</h3>
            <p>{s.body}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

function UseCases() {
  return (
    <section className="section" id="use-cases">
      <div className="section-head">
        <span className="kicker">Use cases</span>
        <h2>Built for back-to-back days.</h2>
      </div>
      <div className="grid grid--3">
        {USE_CASES.map((u) => (
          <article key={u.tag} className="case">
            <span className="case-tag">{u.tag}</span>
            <h3>{u.title}</h3>
            <p>{u.body}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function FAQ() {
  const items = [
    {
      q: "Does Ali record my meetings?",
      a: "Audio is processed locally on your machine for live transcription. Nothing is uploaded unless an action you approve requires it.",
    },
    {
      q: "Do my teammates need to install anything?",
      a: "No. Ali runs only on your laptop. There’s no bot in the call, no join link, nothing for the other side to consent to.",
    },
    {
      q: "What can it actually do?",
      a: "Search and book flights, hold calendar slots, look up docs, draft email/Slack replies, pull up reference material on screen, and queue up multi-step workflows for review.",
    },
    {
      q: "Will it act without asking me?",
      a: "Anything that touches the outside world — sending mail, booking, posting — waits for your one-tap approval at the end of the meeting.",
    },
    {
      q: "When can I use it?",
      a: "We’re onboarding a small group of design partners now. Join the waitlist and we’ll reach out.",
    },
  ];
  return (
    <section className="section section--alt" id="faq">
      <div className="section-head">
        <span className="kicker">FAQ</span>
        <h2>Reasonable questions.</h2>
      </div>
      <div className="faq">
        {items.map((it) => (
          <details key={it.q} className="faq-item">
            <summary>{it.q}</summary>
            <p>{it.a}</p>
          </details>
        ))}
      </div>
    </section>
  );
}

const WAITLIST_ENDPOINT =
  (import.meta.env.VITE_WAITLIST_ENDPOINT as string | undefined) ||
  "https://formspree.io/f/myklvkvz";

function Waitlist() {
  const [email, setEmail] = useState("");
  const [done, setDone] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = email.trim();
    if (!trimmed || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch(WAITLIST_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          email: trimmed,
          source: "landing-waitlist",
          submitted_at: new Date().toISOString(),
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        const msg =
          data?.errors?.[0]?.message ||
          data?.error ||
          `Something went wrong (${res.status}). Try again?`;
        setError(msg);
        return;
      }
      setDone(true);
      setEmail("");
    } catch {
      setError("Couldn't reach the server. Check your connection and try again.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className="cta" id="waitlist">
      <div className="cta-card">
        <h2>Stop running errands inside your meetings.</h2>
        <p>
          Join the early-access list. We’ll send a private install link as we onboard new
          design partners.
        </p>
        {done ? (
          <div className="cta-done">
            You’re on the list. We’ll be in touch shortly.
          </div>
        ) : (
          <form className="cta-form" onSubmit={submit}>
            <input
              type="email"
              required
              placeholder="you@company.com"
              value={email}
              disabled={submitting}
              onChange={(e) => setEmail(e.target.value)}
            />
            <button
              className="btn btn--primary btn--lg"
              type="submit"
              disabled={submitting}
            >
              {submitting ? "Sending…" : "Request access"}
            </button>
          </form>
        )}
        {error && <div className="cta-error">{error}</div>}
        <span className="cta-note">No spam. We email when there’s a spot for you.</span>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div className="footer-brand">
          <span className="logo-text">Ali</span>
          <span className="footer-tag">Meetings, finished.</span>
        </div>
        <div className="footer-cols">
          <div>
            <h4>Product</h4>
            <a href="#features">Features</a>
            <a href="#how">How it works</a>
            <a href="#use-cases">Use cases</a>
          </div>
          <div>
            <h4>Company</h4>
            <a href="#faq">FAQ</a>
            <a href="#waitlist">Early access</a>
            <a href="mailto:hello@askali.app">Contact</a>
          </div>
          <div>
            <h4>Legal</h4>
            <a href="#">Privacy</a>
            <a href="#">Terms</a>
          </div>
        </div>
      </div>
      <div className="footer-bottom">
        © {new Date().getFullYear()} Ali. All rights reserved.
      </div>
    </footer>
  );
}

function App() {
  useEffect(() => {
    document.title = "Ali — the AI that finishes your meetings";
  }, []);

  return (
    <div className="site">
      <NavBar />
      <Hero />
      <Features />
      <HowItWorks />
      <UseCases />
      <FAQ />
      <Waitlist />
      <Footer />
    </div>
  );
}

export default App;
