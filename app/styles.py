"""Refined visual theme for CineScale Streamlit frontend — V3."""

APP_CSS = r"""
<style>
:root {
    --cs-red: #e50914;
    --cs-red-2: #ff4f57;
    --cs-red-soft: #fff2f2;
    --cs-ink: #17181c;
    --cs-muted: #6d7480;
    --cs-line: #e9ebef;
    --cs-bg: #f7f8fa;
    --cs-panel: #ffffff;
    --cs-green: #16a36f;
    --cs-purple: #8b64d9;
    --cs-yellow: #f8b400;
    --cs-shadow: 0 12px 34px rgba(18, 24, 32, 0.06);
}

html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 72% -8%, rgba(229,9,20,.04), transparent 28%),
        linear-gradient(180deg, #ffffff 0%, #fafafb 46%, #f7f8fa 100%);
    color: var(--cs-ink) !important;
}

.block-container {
    max-width: 1520px;
    padding-top: 1rem;
    padding-bottom: 2.2rem;
}

#MainMenu,
footer,
header[data-testid="stHeader"],
div[data-testid="stToolbar"] {
    visibility: hidden !important;
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    overflow: hidden !important;
}

[data-testid="stAppViewContainer"] {
    background: transparent;
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,.97);
    border-right: 1px solid var(--cs-line);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
[data-testid="stSidebar"] * { color: var(--cs-ink); }

.cs-brand {
    display: flex;
    align-items: flex-start;
    gap: .8rem;
    margin: .15rem 0 1.7rem 0;
}
.cs-brand-mark {
    width: 40px;
    height: 40px;
    border-radius: 13px;
    display: grid;
    place-items: center;
    color: #fff;
    background: linear-gradient(135deg, #ff5b62, var(--cs-red));
    box-shadow: 0 9px 22px rgba(229,9,20,.18);
    font-size: 1.08rem;
    margin-top: 2px;
}
.cs-brand-copy { padding-top: 1px; }
.cs-brand-name {
    font-size: 1.62rem;
    line-height: 1;
    font-weight: 850;
    letter-spacing: -.045em;
    color: var(--cs-red) !important;
}
.cs-brand-sub {
    color: var(--cs-muted) !important;
    font-size: .72rem;
    margin-top: .34rem;
    max-width: 205px;
    line-height: 1.45;
}

.cs-kicker {
    color: #717887 !important;
    font-size: .67rem;
    font-weight: 850;
    letter-spacing: .09em;
    text-transform: uppercase;
    margin: 1.2rem 0 .62rem 0;
}
.cs-kicker.first { margin-top: .15rem; }

[data-testid="stSidebar"] [data-testid="stSelectbox"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="combobox"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-rac=""],
[data-testid="stSidebar"] div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border-radius: 13px !important;
    color-scheme: light;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="combobox"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] input[role="combobox"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] button[aria-label="Open"],
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #e1e4e9 !important;
    border-radius: 13px !important;
    min-height: 46px;
    box-shadow: 0 5px 15px rgba(23,28,38,.035);
    color-scheme: light;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] input[role="combobox"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] button[aria-label="Open"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] button[aria-label="Open"] img,
[data-testid="stSidebar"] div[data-baseweb="select"] [data-baseweb="select-value"],
[data-testid="stSidebar"] div[data-baseweb="select"] [data-baseweb="select-single-value"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="option"] {
    color: #1c1d21 !important;
    -webkit-text-fill-color: #1c1d21 !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] input[role="combobox"] {
    background-color: transparent !important;
    caret-color: #1c1d21 !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-rac=""] > * {
    background-color: #ffffff !important;
}
div[data-baseweb="popover"] [role="listbox"],
div[data-baseweb="popover"] [role="listbox"] li,
div[data-baseweb="popover"] [role="listbox"] [role="option"] {
    background-color: #ffffff !important;
    color: #1c1d21 !important;
}
div[data-baseweb="popover"] [role="listbox"] li:hover,
div[data-baseweb="popover"] [role="listbox"] [role="option"]:hover {
    background-color: #fff2f2 !important;
}

.cs-user-card,
.cs-history-row,
.cs-movie-card,
.cs-health-card,
.cs-rec-card,
.cs-stat-card {
    background: rgba(255,255,255,.97);
    border: 1px solid var(--cs-line);
    box-shadow: var(--cs-shadow);
}

.cs-user-card {
    display: flex;
    align-items: center;
    gap: .78rem;
    border-radius: 16px;
    padding: .9rem;
    margin: .8rem 0 .25rem 0;
}
.cs-avatar {
    width: 42px;
    height: 42px;
    flex: 0 0 42px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, #ff6268, var(--cs-red));
    color: #fff !important;
    font-size: 1rem;
}
.cs-user-copy { min-width: 0; }
.cs-user-id {
    color: var(--cs-ink) !important;
    font-size: .98rem;
    font-weight: 820;
}
.cs-user-meta {
    color: var(--cs-muted) !important;
    font-size: .75rem;
    margin-top: .12rem;
}

.cs-stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: .65rem;
}
.cs-stat-card {
    border-radius: 14px;
    padding: .88rem;
    display: flex;
    align-items: center;
    gap: .78rem;
}
.cs-stat-card.wide { grid-column: 1 / -1; }
.cs-stat-icon {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    background: #fff1f2;
    color: var(--cs-red) !important;
    font-size: 1.1rem;
    font-weight: 800;
}
.cs-stat-icon.green { background: #eaf8f3; color: var(--cs-green) !important; }
.cs-stat-icon.purple { background: #f4eefe; color: var(--cs-purple) !important; }
.cs-stat-label {
    color: #777e8b !important;
    font-size: .68rem;
    font-weight: 750;
}
.cs-stat-value,
.cs-affinity-value {
    margin-top: .27rem;
    font-size: 1.22rem;
    line-height: 1.15;
    font-weight: 850;
    color: var(--cs-ink) !important;
}
.cs-stat-value.red { color: var(--cs-red) !important; }
.cs-stat-value.green { color: var(--cs-green) !important; }
.cs-affinity-value { font-size: 1.05rem; color: #23252a !important; }

.cs-genre-item { margin: .76rem 0; }
.cs-genre-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: .34rem;
    color: #3c414a !important;
    font-size: .77rem;
}
.cs-track {
    width: 100%;
    height: 6px;
    border-radius: 999px;
    overflow: hidden;
    background: #eceef1;
}
.cs-track.large { height: 7px; }
.cs-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #ff535b, var(--cs-red));
}
.cs-fill.g2 { background: linear-gradient(90deg, #ff9c42, #ff6b35); }
.cs-fill.g3 { background: linear-gradient(90deg, #facc15, #f59e0b); }
.cs-fill.g4 { background: linear-gradient(90deg, #57a9df, #2784c5); }
.cs-fill.g5 { background: linear-gradient(90deg, #9c7be8, #7653ca); }

.cs-note {
    margin-top: 1rem;
    padding: .75rem .82rem;
    border: 1px solid #ececf2;
    border-radius: 12px;
    background: #fafafd;
    color: var(--cs-muted) !important;
    font-size: .72rem;
    line-height: 1.45;
}

/* Top nav */
[data-testid="stPills"] {
    margin-bottom: .2rem;
}
[data-testid="stPills"] [role="tablist"] {
    gap: 1.2rem;
}
[data-testid="stPills"] button,
[data-testid="stPills"] [role="option"] {
    background: transparent !important;
    color: #1e2025 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: .3rem 0 .65rem 0 !important;
    font-weight: 800 !important;
    box-shadow: none !important;
}
[data-testid="stPills"] [aria-selected="true"],
[data-testid="stPills"] button[aria-pressed="true"] {
    color: var(--cs-red) !important;
    border-bottom: 2px solid var(--cs-red) !important;
}

.stTextInput input {
    background: #fff !important;
    color: var(--cs-ink) !important;
    border: 1px solid #e1e4e8 !important;
    border-radius: 13px !important;
    min-height: 46px;
    box-shadow: 0 6px 16px rgba(23,28,38,.035);
}
.stTextInput input:focus {
    border-color: #ff9ca1 !important;
    box-shadow: 0 0 0 2px rgba(229,9,20,.06) !important;
}

.cs-demo-alert {
    display: inline-flex;
    align-items: center;
    gap: .75rem;
    border: 1px solid #f6d8db;
    background: linear-gradient(90deg, #fff8f8, #fffdfd);
    border-radius: 14px;
    padding: .7rem .9rem;
    margin: .35rem 0 1rem 0;
}
.cs-demo-badge {
    background: var(--cs-red-soft);
    border: 1px solid #ffd6d9;
    color: var(--cs-red) !important;
    border-radius: 999px;
    padding: .28rem .6rem;
    font-size: .68rem;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: .04em;
    white-space: nowrap;
}
.cs-demo-text {
    color: var(--cs-muted) !important;
    font-size: .78rem;
}

/* Main content */
.cs-page-title {
    color: var(--cs-ink) !important;
    font-size: clamp(2rem, 2.5vw, 2.8rem);
    line-height: 1.04;
    font-weight: 880;
    letter-spacing: -.052em;
    margin: .22rem 0 .35rem 0;
}
.cs-page-title span { font-size: .82em; }
.cs-page-subtitle {
    color: var(--cs-muted) !important;
    font-size: .96rem;
    margin-bottom: 1.35rem;
}

.cs-section-head-wrap {
    margin: .35rem 0 .9rem 0;
}
.cs-section-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: .75rem;
}
.cs-section-head {
    display: flex;
    align-items: center;
    gap: .58rem;
}
.cs-section-accent {
    width: 4px;
    height: 22px;
    border-radius: 999px;
    background: linear-gradient(180deg, #ff5960, var(--cs-red));
}
.cs-section-title {
    color: var(--cs-ink) !important;
    font-size: 1.06rem;
    font-weight: 830;
}
.cs-section-subtitle {
    color: var(--cs-muted) !important;
    font-size: .76rem;
    margin: .3rem 0 0 .62rem;
}
.cs-section-action {
    color: var(--cs-red) !important;
    border: 1px solid #f6d8db;
    background: #fffafa;
    border-radius: 12px;
    padding: .45rem .8rem;
    font-size: .74rem;
    font-weight: 800;
    white-space: nowrap;
}
.cs-grid-spacer { height: .75rem; }

.cs-rec-card {
    height: 100%;
    border-radius: 18px;
    padding: .8rem;
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
.cs-rec-card:hover {
    transform: translateY(-3px);
    border-color: #f3c9cc;
    box-shadow: 0 16px 38px rgba(23,28,38,.10);
}
.cs-rec-poster {
    position: relative;
    width: 100%;
    height: 112px;
    border-radius: 14px;
    background: linear-gradient(145deg, #f3f4f6, #e7e9ee);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: .8rem;
    overflow: hidden;
}
.cs-rec-poster .cs-movie-symbol {
    width: auto;
    height: auto;
    background: none;
    font-size: 2.1rem;
    color: #b9bfc9 !important;
}
.cs-rec-poster .cs-match {
    position: absolute;
    top: 8px;
    right: 8px;
}
.cs-movie-symbol {
    width: 40px;
    height: 40px;
    display: grid;
    place-items: center;
    border-radius: 12px;
    background: #f4f5f7;
    color: #8b929d !important;
    font-size: 1rem;
}
.cs-match {
    border: 1px solid #ffd2d5;
    background: var(--cs-red-soft);
    color: var(--cs-red) !important;
    border-radius: 999px;
    padding: .32rem .56rem;
    font-size: .7rem;
    font-weight: 850;
    white-space: nowrap;
}
.cs-rec-body {
    display: flex;
    flex-direction: column;
    flex: 1;
}
.cs-rec-title {
    color: var(--cs-ink) !important;
    font-size: .96rem;
    font-weight: 840;
    line-height: 1.25;
    min-height: 2.45em;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.cs-rec-year {
    color: #7b818d !important;
    font-size: .73rem;
    margin-top: .35rem;
    font-weight: 700;
}
.cs-rec-meta {
    color: var(--cs-muted) !important;
    font-size: .72rem;
    line-height: 1.42;
    margin-top: .26rem;
}
.cs-rec-rating {
    display: flex;
    align-items: center;
    gap: .35rem;
    margin-top: auto;
    padding-top: .6rem;
    font-size: .84rem;
    font-weight: 830;
    color: var(--cs-ink) !important;
}
.cs-rec-rating .cs-rec-star {
    color: var(--cs-yellow) !important;
    font-size: .88rem;
}

.cs-history-row {
    display: grid;
    grid-template-columns: 38px 1fr auto;
    align-items: center;
    gap: .75rem;
    border-radius: 13px;
    padding: .68rem .75rem;
    margin-bottom: .5rem;
    box-shadow: 0 5px 18px rgba(23,28,38,.045);
}
.cs-history-icon {
    width: 34px;
    height: 34px;
    display: grid;
    place-items: center;
    border-radius: 9px;
    background: #f4f5f7;
    color: #9ba1aa !important;
    font-size: .9rem;
}
.cs-history-main { min-width: 0; }
.cs-history-title {
    color: var(--cs-ink) !important;
    font-size: .86rem;
    font-weight: 780;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.cs-history-meta {
    color: var(--cs-muted) !important;
    font-size: .7rem;
    margin-top: .13rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.cs-rating {
    color: var(--cs-red) !important;
    font-weight: 850;
    font-size: .82rem;
    white-space: nowrap;
}

.cs-taste-row {
    padding: .42rem .08rem .62rem .08rem;
    border-bottom: 1px solid #f1f2f4;
}
.cs-taste-row:last-of-type { border-bottom: none; }
.cs-taste-line {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: .42rem;
    font-size: .8rem;
}
.cs-taste-line strong { color: #444952 !important; }
.cs-taste-name {
    color: var(--cs-ink) !important;
    font-weight: 720;
    display: flex;
    align-items: center;
    gap: .46rem;
}
.cs-genre-bullet {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--cs-red);
}
.cs-genre-bullet.g2 { background: #ff6b35; }
.cs-genre-bullet.g3 { background: #f59e0b; }
.cs-genre-bullet.g4 { background: #2784c5; }
.cs-genre-bullet.g5 { background: #7653ca; }

.cs-soft-note {
    border: 1px solid #f0e5e7;
    background: linear-gradient(90deg, #fff8f8, #fffdfd);
    border-radius: 12px;
    padding: .68rem .8rem;
    margin-top: .75rem;
    color: #79717a !important;
    font-size: .71rem;
    line-height: 1.4;
}
.cs-soft-note.wide { max-width: 920px; }

/* Genre donut chart */
.cs-donut-row {
    display: flex;
    align-items: center;
    gap: 1.7rem;
    padding: .3rem 0 .1rem 0;
}
.cs-donut {
    width: 152px;
    height: 152px;
    flex: 0 0 152px;
    border-radius: 50%;
    position: relative;
    box-shadow: 0 10px 26px rgba(23,28,38,.08);
}
.cs-donut-hole {
    position: absolute;
    inset: 23%;
    background: var(--cs-panel);
    border-radius: 50%;
}
.cs-donut-legend {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: .62rem;
    min-width: 0;
}
.cs-donut-legend-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: .6rem;
    font-size: .85rem;
}
.cs-donut-legend-name {
    display: flex;
    align-items: center;
    gap: .55rem;
    color: var(--cs-ink) !important;
    font-weight: 680;
}
.cs-donut-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex: 0 0 10px;
}
.cs-donut-legend-pct {
    color: var(--cs-muted) !important;
    font-weight: 750;
    white-space: nowrap;
}
.cs-genre-tip {
    display: flex;
    align-items: flex-start;
    gap: .55rem;
    margin-top: 1.1rem;
    padding: .75rem .85rem;
    border: 1px solid #ffd6d9;
    background: var(--cs-red-soft);
    border-radius: 12px;
    color: #8a4a4a !important;
    font-size: .74rem;
    line-height: 1.45;
}

.cs-cold {
    display: flex;
    align-items: center;
    gap: .8rem;
    border: 1px solid #f6dda6;
    background: #fff9e9;
    border-radius: 14px;
    padding: .9rem 1rem;
    color: #6b5316 !important;
    margin: 1rem 0 .2rem 0;
}
.cs-cold strong { color: #3c300f !important; }
.cs-cold-icon { font-size: 1.25rem; }

.cs-movie-card {
    display: flex;
    align-items: center;
    gap: .82rem;
    border-radius: 15px;
    padding: .85rem;
    min-height: 112px;
    margin-bottom: .72rem;
}
.cs-movie-poster {
    width: 54px;
    height: 72px;
    flex: 0 0 54px;
    display: grid;
    place-items: center;
    border-radius: 10px;
    background: linear-gradient(145deg, #f7f7f9, #eceef2);
    color: #9ba1aa !important;
}
.cs-movie-copy { min-width: 0; }
.cs-movie-title {
    color: var(--cs-ink) !important;
    font-weight: 810;
    line-height: 1.25;
}
.cs-movie-meta {
    color: var(--cs-muted) !important;
    font-size: .73rem;
    margin-top: .3rem;
    line-height: 1.4;
}

.cs-health-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: .85rem;
    margin: .3rem 0 1.2rem 0;
}
.cs-health-card {
    display: flex;
    align-items: center;
    gap: .85rem;
    border-radius: 16px;
    padding: 1rem;
}
.cs-health-icon {
    width: 42px;
    height: 42px;
    display: grid;
    place-items: center;
    border-radius: 12px;
    background: #f7f7f9;
    font-size: 1rem;
}
.cs-health-label { color: var(--cs-muted) !important; font-size: .73rem; }
.cs-health-value { color: var(--cs-ink) !important; font-size: 1.5rem; font-weight: 870; margin-top: .16rem; }
.cs-live-ok {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    color: #24775b !important;
    background: #f2fbf7;
    border: 1px solid #d9f1e7;
    border-radius: 999px;
    padding: .48rem .75rem;
    font-size: .72rem;
    font-weight: 760;
    margin-bottom: .9rem;
}
.cs-live-ok span {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--cs-green);
}

.stButton > button {
    border-radius: 11px !important;
    font-weight: 760 !important;
}
[data-testid="stToggle"] label p { color: #50555e !important; font-size: .75rem !important; }

/* "Why This?" popover trigger */
[data-testid="stPopover"] > button {
    margin-top: .45rem;
    border-radius: 10px !important;
    font-size: .73rem !important;
    font-weight: 750 !important;
    color: var(--cs-red) !important;
    border: 1px solid #f6d8db !important;
    background: #fffafa !important;
    padding: .3rem .7rem !important;
}
[data-testid="stPopover"] > button:hover {
    background: var(--cs-red-soft) !important;
    border-color: #f3c9cc !important;
}

@media (max-width: 1050px) {
    .cs-health-grid { grid-template-columns: 1fr; }
    .block-container { padding-left: .85rem; padding-right: .85rem; }
    .cs-section-row { flex-wrap: wrap; }
}
</style>
"""
