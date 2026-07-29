"""CineScale Streamlit frontend — polished V3.

V3 focuses on UI refinement so the frontend visually matches the approved mockup
more closely while still using only backend-supported fields.
"""

from __future__ import annotations

import html
import re
from decimal import Decimal
from typing import Dict, Iterable, List, Tuple

import streamlit as st

from db_queries import (
    benchmark_recommendation,
    get_all_genres,
    get_all_users,
    get_explanation,
    get_genre_distribution,
    get_rating_count,
    get_recommendations,
    get_system_health,
    get_user_history,
    get_user_stats,
    search_movies,
)
from styles import APP_CSS


st.set_page_config(
    page_title="CineScale",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(APP_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def safe_text(value: object) -> str:
    return html.escape("" if value is None else str(value))


def render_html(markup: str, container=None) -> None:
    """Render an HTML fragment safely.

    Streamlit's markdown parser treats any blank line (including a line that
    is only whitespace, which happens whenever an interpolated placeholder is
    empty) as the end of an HTML block. Anything indented that follows is
    then rendered as a literal code block instead of HTML. Stripping each
    line and dropping empty ones guarantees the whole fragment stays one
    continuous HTML block regardless of Python indentation or empty
    placeholders.
    """
    target = container if container is not None else st
    cleaned_lines = [line.strip() for line in markup.splitlines()]
    cleaned = "\n".join(line for line in cleaned_lines if line)
    target.markdown(cleaned, unsafe_allow_html=True)


def to_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_number(value: object) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def split_title_year(raw_title: str) -> Tuple[str, str]:
    match = re.match(r"^(.*)\s+\((\d{4})\)$", raw_title or "")
    if match:
        return match.group(1), match.group(2)
    return raw_title or "Untitled", ""


def genre_text(raw_genres: object, max_items: int = 3) -> str:
    if not raw_genres:
        return "Genres unavailable"
    items = [
        genre
        for genre in str(raw_genres).split("|")
        if genre and genre != "(no genres listed)"
    ]
    return ", ".join(items[:max_items]) if items else "Genres unavailable"


# ---------------------------------------------------------------------------
# Cached live backend calls
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def live_users() -> List[Dict]:
    return get_all_users(limit=500)


@st.cache_data(ttl=60, show_spinner=False)
def live_user_bundle(user_id: int) -> Tuple[Dict, List[Dict], int, List[Dict]]:
    return (
        get_user_stats(user_id),
        get_genre_distribution(user_id),
        get_rating_count(user_id),
        get_user_history(user_id, limit=50),
    )


@st.cache_data(ttl=45, show_spinner=False)
def live_recommendations(user_id: int, top_n: int) -> List[Dict]:
    return get_recommendations(user_id, top_n=top_n)


@st.cache_data(ttl=60, show_spinner=False)
def live_health() -> Dict:
    return get_system_health()


@st.cache_data(ttl=30, show_spinner=False)
def live_movie_search(query: str, limit: int = 30, genre: str | None = None) -> List[Dict]:
    return search_movies(query, limit=limit, genre=genre)


@st.cache_data(ttl=300, show_spinner=False)
def live_all_genres() -> List[str]:
    return get_all_genres()


@st.cache_data(ttl=60, show_spinner=False)
def live_explanation(user_id: int, movie_id: int) -> List[Dict]:
    return get_explanation(user_id, movie_id)


# ---------------------------------------------------------------------------
# Live data source selection
# ---------------------------------------------------------------------------

LIVE_ERROR: str | None = None
try:
    USERS = live_users()
    if not USERS:
        raise RuntimeError("No users with embeddings were returned from the live database.")
except Exception as exc:
    LIVE_ERROR = str(exc)
    USERS = []


def load_user_bundle(user_id: int):
    return live_user_bundle(user_id)


def load_recommendations(user_id: int, top_n: int):
    return live_recommendations(user_id, top_n)


def load_health():
    return live_health()


def load_movie_search(query: str, limit: int = 30, genre: str | None = None):
    return live_movie_search(query, limit, genre=genre)


# ---------------------------------------------------------------------------
# Reusable UI components
# ---------------------------------------------------------------------------

def render_brand() -> None:
    render_html(
        """
        <div class="cs-brand">
            <div class="cs-brand-mark">🎬</div>
            <div class="cs-brand-copy">
                <div class="cs-brand-name">CineScale</div>
                <div class="cs-brand-sub">Smart recommendations, just for you.</div>
            </div>
        </div>
        """,
        container=st.sidebar,
    )


def render_mode_strip() -> None:
    return


def render_sidebar_profile(user_id: int, rating_count: int, stats: Dict, genres: List[Dict]) -> None:
    avg = stats.get("avg_score")
    avg_text = f"{to_float(avg):.1f} / 5" if avg is not None else "—"
    top_affinity = safe_text(stats.get("top_affinity") or "Not available")

    render_html(
        f"""
        <div class="cs-user-card">
            <div class="cs-avatar">👤</div>
            <div class="cs-user-copy">
                <div class="cs-user-id">User {user_id}</div>
                <div class="cs-user-meta">{format_number(rating_count)} ratings</div>
            </div>
        </div>
        <div class="cs-kicker">User Overview</div>
        <div class="cs-stat-grid">
            <div class="cs-stat-card">
                <div class="cs-stat-icon">★</div>
                <div>
                    <div class="cs-stat-label">Ratings</div>
                    <div class="cs-stat-value red">{format_number(stats.get('ratings_count', 0))}</div>
                </div>
            </div>
            <div class="cs-stat-card">
                <div class="cs-stat-icon green">↗</div>
                <div>
                    <div class="cs-stat-label">Average Rating</div>
                    <div class="cs-stat-value green">{avg_text}</div>
                </div>
            </div>
            <div class="cs-stat-card wide">
                <div class="cs-stat-icon purple">♥</div>
                <div>
                    <div class="cs-stat-label">Top Affinity</div>
                    <div class="cs-affinity-value">{top_affinity}</div>
                </div>
            </div>
        </div>
        """,
        container=st.sidebar,
    )

    st.sidebar.markdown('<div class="cs-kicker">Genre Preferences</div>', unsafe_allow_html=True)
    if not genres:
        st.sidebar.caption("No genre preference data yet.")
    else:
        for item in genres[:5]:
            pct = max(0, min(100, int(to_float(item.get("percentage")))))
            render_html(
                f"""
                <div class="cs-genre-item">
                    <div class="cs-genre-head">
                        <span>{safe_text(item.get('genre'))}</span><span>{pct}%</span>
                    </div>
                    <div class="cs-track"><div class="cs-fill" style="width:{pct}%"></div></div>
                </div>
                """,
                container=st.sidebar,
            )

    render_html(
        '<div class="cs-note">ⓘ Recommendations are based on your ratings and similar users.</div>',
        container=st.sidebar,
    )


def section_heading(title: str, subtitle: str | None = None, action_text: str | None = None) -> None:
    action_html = f'<div class="cs-section-action">{safe_text(action_text)}</div>' if action_text else ""
    subtitle_html = f'<div class="cs-section-subtitle">{safe_text(subtitle)}</div>' if subtitle else ""
    render_html(
        f"""
        <div class="cs-section-head-wrap">
            <div class="cs-section-row">
                <div class="cs-section-head">
                    <span class="cs-section-accent"></span>
                    <span class="cs-section-title">{safe_text(title)}</span>
                </div>
                {action_html}
            </div>
            {subtitle_html}
        </div>
        """
    )


def render_recommendation_card(rec: Dict, user_id: int) -> None:
    title, year = split_title_year(str(rec.get("title") or "Untitled"))
    genres = genre_text(rec.get("genres"), 3)
    similarity = max(0.0, min(1.0, to_float(rec.get("similarity"))))
    match_pct = round(similarity * 100)
    movie_id = rec.get("movie_id")

    render_html(
        f"""
        <div class="cs-rec-card">
            <div class="cs-rec-poster">
                <div class="cs-movie-symbol">🎬</div>
                <div class="cs-match">{match_pct}%</div>
            </div>
            <div class="cs-rec-body">
                <div class="cs-rec-title">{safe_text(title)}</div>
                <div class="cs-rec-year">{safe_text(year or '—')}</div>
                <div class="cs-rec-meta">{safe_text(genres)}</div>
            </div>
        </div>
        """
    )

    if movie_id:
        with st.popover("Why This?"):
            explanations = live_explanation(user_id, int(movie_id))
            if not explanations:
                st.caption("No explanation available.")
            else:
                st.markdown("**Because you enjoyed these movies:**")
                for item in explanations:
                    shared = ", ".join(item.get("shared_genres") or [])
                    st.markdown(
                        f"- **{safe_text(item['title'])}** — ★ {to_float(item['rating']):.1f}"
                        + (f" _(shared: {safe_text(shared)})_" if shared else "")
                    )


def render_recommendations(recommendations: List[Dict], user_id: int) -> None:
    if not recommendations:
        st.info("No recommendations are available for this user.")
        return

    visible = recommendations[:5]
    columns = st.columns(len(visible), gap="small")
    for index, rec in enumerate(visible):
        with columns[index]:
            render_recommendation_card(rec, user_id)

    remaining = recommendations[5:10]
    if remaining:
        show_more = st.toggle("Show 5 more recommendations", value=False)
        if show_more:
            columns_2 = st.columns(len(remaining), gap="small")
            for index, rec in enumerate(remaining):
                with columns_2[index]:
                    render_recommendation_card(rec, user_id)


def render_history(history: Iterable[Dict], per_page: int = 5) -> None:
    rows = list(history)
    if not rows:
        st.info("No rating history is available for this user.")
        return

    total = len(rows)
    total_pages = max(1, (total + per_page - 1) // per_page)

    page_key = "history_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    page = st.session_state[page_key]
    start = page * per_page
    page_rows = rows[start:start + per_page]

    for row in page_rows:
        title, year = split_title_year(str(row.get("title") or "Untitled"))
        meta = " • ".join(part for part in [year, genre_text(row.get("genres"), 2)] if part)
        rating = to_float(row.get("rating"))
        render_html(
            f"""
            <div class="cs-history-row">
                <div class="cs-history-icon">🎞️</div>
                <div class="cs-history-main">
                    <div class="cs-history-title">{safe_text(title)}</div>
                    <div class="cs-history-meta">{safe_text(meta)}</div>
                </div>
                <div class="cs-rating">★ {rating:.1f}</div>
            </div>
            """
        )

    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("← Previous", disabled=(page == 0), key="hist_prev"):
                st.session_state[page_key] = page - 1
                st.rerun()
        with col2:
            st.markdown(
                f'<div style="text-align:center;font-size:.8rem;color:#6d7480;padding-top:.4rem">'
                f'Page {page + 1} of {total_pages}</div>',
                unsafe_allow_html=True,
            )
        with col3:
            if st.button("Next →", disabled=(page >= total_pages - 1), key="hist_next"):
                st.session_state[page_key] = page + 1
                st.rerun()


DONUT_COLORS = ["#e50914", "#ff8a3d", "#f8b400", "#2784c5", "#8b64d9"]


def render_taste(genres: Iterable[Dict]) -> None:
    rows = list(genres)[:5]
    if not rows:
        st.info("No genre preference data is available for this user.")
        return

    shares = [max(0.0, to_float(row.get("percentage"))) for row in rows]
    total = sum(shares) or 1.0

    gradient_parts = []
    start = 0.0
    for index, share in enumerate(shares):
        end = start + (share / total) * 360.0
        color = DONUT_COLORS[index % len(DONUT_COLORS)]
        gradient_parts.append(f"{color} {start:.2f}deg {end:.2f}deg")
        start = end
    gradient = ", ".join(gradient_parts)

    legend_html = "".join(
        f'<div class="cs-donut-legend-row">'
        f'<div class="cs-donut-legend-name">'
        f'<span class="cs-donut-dot" style="background:{DONUT_COLORS[index % len(DONUT_COLORS)]}"></span>'
        f'{safe_text(row.get("genre"))}'
        f'</div>'
        f'<div class="cs-donut-legend-pct">{max(0, min(100, int(to_float(row.get("percentage")))))}%</div>'
        f'</div>'
        for index, row in enumerate(rows)
    )

    render_html(
        f"""
        <div class="cs-donut-row">
            <div class="cs-donut" style="background:conic-gradient({gradient});">
                <div class="cs-donut-hole"></div>
            </div>
            <div class="cs-donut-legend">{legend_html}</div>
        </div>
        <div class="cs-genre-tip">
            <span>💡</span>
            <span>Percentages show how much you prefer each genre compared to other users.</span>
        </div>
        """
    )


def render_movie_cards(movies: List[Dict]) -> None:
    if not movies:
        st.info("No movies matched your search.")
        return

    columns = st.columns(3, gap="medium")
    for index, movie in enumerate(movies):
        title, year = split_title_year(str(movie.get("title") or "Untitled"))
        meta = " • ".join(part for part in [year, genre_text(movie.get("genres"), 3)] if part)
        with columns[index % 3]:
            render_html(
                f"""
                <div class="cs-movie-card">
                    <div class="cs-movie-poster">🎬</div>
                    <div class="cs-movie-copy">
                        <div class="cs-movie-title">{safe_text(title)}</div>
                        <div class="cs-movie-meta">{safe_text(meta)}</div>
                    </div>
                </div>
                """
            )


def render_health_card(label: str, value: object, icon: str) -> str:
    return (
        f'<div class="cs-health-card">'
        f'<div class="cs-health-icon">{icon}</div>'
        f'<div>'
        f'<div class="cs-health-label">{safe_text(label)}</div>'
        f'<div class="cs-health-value">{format_number(value)}</div>'
        f'</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Sidebar + navigation
# ---------------------------------------------------------------------------

render_brand()

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] [data-testid="stSelectbox"] input[role="combobox"],
    [data-testid="stSidebar"] [data-testid="stSelectbox"] button[aria-label="Open"],
    [data-testid="stSidebar"] [data-testid="stSelectbox"] button[aria-label="Open"] img,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] button[aria-label="Open"] svg,
    div[data-baseweb="popover"] [role="listbox"] [role="option"],
    div[data-baseweb="popover"] [role="listbox"] [role="option"] * {
        background-color: #ffffff !important;
        color: #1c1d21 !important;
        fill: #1c1d21 !important;
        -webkit-text-fill-color: #1c1d21 !important;
    }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] input[role="combobox"] {
        caret-color: #1c1d21 !important;
    }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] button[aria-label="Open"] svg {
        display: none !important;
    }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] button[aria-label="Open"]::after {
        content: "⌄";
        display: block;
        color: #1c1d21;
        font-size: 1.25rem;
        font-weight: 700;
        line-height: 1;
        margin-top: -0.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not USERS:
    st.error("CineScale could not load live user embeddings from PostgreSQL.")
    if LIVE_ERROR:
        with st.expander("Technical details"):
            st.code(LIVE_ERROR)
    st.stop()

user_ids = [int(row["user_id"]) for row in USERS]
rating_count_by_user = {
    int(row["user_id"]): int(row.get("rating_count") or 0)
    for row in USERS
}

st.sidebar.markdown('<div class="cs-kicker first">Select User</div>', unsafe_allow_html=True)
selected_user = st.sidebar.selectbox(
    "Select User",
    user_ids,
    format_func=lambda value: f"User {value}  ·  {rating_count_by_user.get(value, 0)} ratings",
    label_visibility="collapsed",
)

if "last_user" not in st.session_state or st.session_state["last_user"] != selected_user:
    st.session_state["history_page"] = 0
    st.session_state["last_user"] = selected_user

try:
    user_stats, genre_distribution, rating_count, user_history = load_user_bundle(selected_user)
except Exception as exc:
    st.error("User data could not be loaded.")
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

render_sidebar_profile(selected_user, rating_count, user_stats, genre_distribution)

nav_col, search_col = st.columns([2.6, 1.4], vertical_alignment="center")
with nav_col:
    navigation = st.pills(
        "Navigation",
        ["For You", "Browse Movies", "System Health"],
        default="For You",
        selection_mode="single",
        label_visibility="collapsed",
    )
with search_col:
    global_search = st.text_input(
        "Search",
        placeholder="🔍  Search movies...",
        label_visibility="collapsed",
        key="global_search",
    ).strip()

if navigation is None:
    navigation = "For You"

render_mode_strip()


# ---------------------------------------------------------------------------
# For You
# ---------------------------------------------------------------------------

if navigation == "For You":
    st.markdown(
        f'<div class="cs-page-title">For You, User {selected_user} <span>👋</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cs-page-subtitle">Personalized picks based on your taste</div>',
        unsafe_allow_html=True,
    )

    section_heading("Recommended For You", action_text="See all")
    recommendations = load_recommendations(selected_user, 10)
    render_recommendations(recommendations, selected_user)

    st.markdown('<div class="cs-grid-spacer"></div>', unsafe_allow_html=True)
    history_col, taste_col = st.columns([1.05, 0.95], gap="large")

    with history_col:
        section_heading("Your Recent Ratings")
        render_history(user_history)

    with taste_col:
        section_heading("Your Taste in Genres")
        render_taste(genre_distribution)

    if rating_count < 5:
        render_html(
            """
            <div class="cs-cold">
                <div class="cs-cold-icon">⚠️</div>
                <div><strong>Not enough ratings yet</strong><br>
                You need at least 5 ratings to get personalized recommendations.</div>
            </div>
            """
        )


# ---------------------------------------------------------------------------
# Browse Movies
# ---------------------------------------------------------------------------

elif navigation == "Browse Movies":
    st.markdown('<div class="cs-page-title">Browse Movies</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cs-page-subtitle">Search the movie catalog already supported by the CineScale backend.</div>',
        unsafe_allow_html=True,
    )

    genres = live_all_genres()
    selected_genre = st.pills(
        "Filter by genre",
        genres,
        selection_mode="single",
        label_visibility="collapsed",
    )

    query = global_search
    results = load_movie_search(query, limit=30, genre=selected_genre)
    heading = "Search Results" if query else ("Genre: " + selected_genre if selected_genre else "Catalog Preview")
    section_heading(heading)
    render_movie_cards(results)


# ---------------------------------------------------------------------------
# System Health
# ---------------------------------------------------------------------------

else:
    st.markdown('<div class="cs-page-title">System Health</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cs-page-subtitle">Live database counts and recommendation-query latency.</div>',
        unsafe_allow_html=True,
    )

    health = load_health()
    render_html(
        '<div class="cs-health-grid">'
        + render_health_card("Users with embeddings", health.get("total_users"), "👤")
        + render_health_card("Movies with embeddings", health.get("total_movies"), "🎬")
        + render_health_card("Stored ratings", health.get("total_ratings"), "★")
        + '</div>'
    )

    render_html(
        '<div class="cs-live-ok"><span></span> PostgreSQL / pgvector connection is active</div>'
    )
    section_heading("Recommendation Query Benchmark")
    runs = st.select_slider("Benchmark runs", options=[10, 25, 50, 100], value=25)
    if st.button("Run benchmark", type="primary"):
        with st.spinner("Running recommendation benchmark..."):
            result = benchmark_recommendation(selected_user, runs=int(runs))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average", f"{result['avg_ms']:.2f} ms")
        c2.metric("p50", f"{result['p50_ms']:.2f} ms")
        c3.metric("p95", f"{result['p95_ms']:.2f} ms")
        c4.metric("p99", f"{result['p99_ms']:.2f} ms")
