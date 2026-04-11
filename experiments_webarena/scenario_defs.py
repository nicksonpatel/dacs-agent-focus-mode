"""Scenario definitions for the DACS WebArena evaluation experiment.

Each agent wraps a real WebArena task (from the public 812-task dataset)
adapted for LLM simulation.  Because the WebArena Docker environment is not
required, each agent receives a short "simulated web page excerpt" embedded
in its system prompt — enough context for the LLM to answer the task and
generate genuine steering questions without browser access.

Tasks are chosen exclusively from WebArena's *informational* subset
(``eval_type == string_match``) so that ground-truth answers are objective
and do not require state-mutating browser actions.

Reference answers come verbatim from the WebArena ``test.raw.json`` file:
  https://github.com/web-arena-x/webarena

Cite: Zhou et al., "WebArena: A Realistic Web Environment for Building
Autonomous Agents", NeurIPS 2024.

Design constraint (same as ``experiments_real_agent``):
    ``decision_hints`` names the *type* of ambiguity without prescribing the
    correct answer.  Only ``DecisionRubric.correct_keywords`` and
    ``judge_context`` carry the expected answer, and those are used exclusively
    by the offline judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Shared dataclasses (same schema as experiments_real_agent.scenario_defs)
# ---------------------------------------------------------------------------

@dataclass
class DecisionRubric:
    """Ground-truth rubric for one expected decision point.

    Attributes:
        topic:            Short label used in analysis tables.
        correct_keywords: Case-insensitive substrings that must appear in the
                          orchestrator response for keyword_score to be 1.
        judge_context:    Plain-text description of what a *correct* orchestrator
                          response looks like — injected into the LLM judge prompt.
        webarena_eval:    Eval type from WebArena (``exact_match``, ``must_include``,
                          ``fuzzy_match``).  Used by the judge for semantic scoring.
        reference_answer: Verbatim text from WebArena ``reference_answers`` field.
    """

    topic: str
    correct_keywords: list[str]
    judge_context: str
    webarena_eval: str = "must_include"
    reference_answer: str = ""


@dataclass
class WebArenaAgentSpec:
    """Specification for one WebArena-backed LLM agent.

    Attributes:
        agent_id:              Unique identifier ("a1", "a2", …).
        webarena_task_id:      Numeric task_id from WebArena ``test.raw.json``.
        webarena_site:         Site name (gitlab, map, shopping_admin, reddit, …).
        task_description:      ≤60-token task description (injected into system
                               prompt and registry; written in simulation-aware
                               framing — "Based on the following web page excerpt …").
        simulated_page_context: Short excerpt (3–8 lines) from the relevant web
                               page that makes the task answerable without a browser.
        decision_hints:        Paragraph naming decision ambiguities the agent
                               should ask the orchestrator about.
        rubrics:               Ordered list of expected decision points.
    """

    agent_id: str
    webarena_task_id: int
    webarena_site: str
    task_description: str
    simulated_page_context: str
    decision_hints: str
    rubrics: list[DecisionRubric] = field(default_factory=list)


@dataclass
class WebArenaScenario:
    """Top-level scenario container."""

    scenario_id: str
    description: str
    agents: list[WebArenaAgentSpec]


# ---------------------------------------------------------------------------
# Simulated page excerpts (factual data, independent of WebArena Docker env)
# ---------------------------------------------------------------------------
#
# These excerpts are constructed from publicly available information that
# mirrors the kind of data the WebArena environments expose.  They are
# intentionally minimal — just enough for the agent to reason through the task.

_PAGE_GITLAB_311 = """\
[Simulated GitLab page: pytorch-GAN / Contributors]
Contributor | Commits (main branch)
Erik Linder-Norén  | 47
V13Axel            | 8
Fei Shen           | 3
(Showing contributions to default branch only; merge commits are counted.)
"""

_PAGE_MAP_52 = """\
[Simulated OpenStreetMap routing result]
Start: Carnegie Mellon University, Pittsburgh, PA
End:   Starbucks, 417 South Craig Street, Pittsburgh, PA
Walking route (normal pace ~4 km/h): 7 min  (0.5 km)
Driving route: 2 min  (0.6 km)
Cycling route: 3 min  (0.5 km)
"""

_PAGE_SHOPPING_279 = """\
[Simulated Shopping Admin — Sales Report Q1 2022]
Product Type       | Units Sold | Revenue
Yoga ball          | 1,842      | $55,260
Resistance band    |   734      | $14,680
Jump rope          |   512      |  $7,680
Foam roller        |   401      |  $9,224
(Quarter 1 2022: Jan–Mar. Ranked by units sold.)
"""

_PAGE_REDDIT_67 = """\
[Simulated Reddit — r/books — Top 10 posts (by upvotes)]
Rank | Title                                                       | Type
1    | "A Christmas Carol changed my perspective on charity"       | Single-book rec
2    | "Finished The Hobbit for the first time — breathtaking"    | Single-book rec
3    | "Best sci-fi of 2023? Drop your suggestions"               | Multi-rec request
4    | "Why I stopped reading literary fiction"                   | Discussion
5    | "Just finished Foundation trilogy — is it worth continuing?" | Multi-book
6    | "Book haul — 10 recommendations for winter"                 | Multi-rec
7    | "The Midnight Library was meh — fight me"                  | Single-book
8    | "Reading challenge: which books cross genre lines?"        | Discussion
9    | "My top 5 nonfiction picks this year"                      | Multi-rec
10   | "Sapiens: still worth it in 2023?"                         | Single-book
"""

_PAGE_GITLAB_312 = """\
[Simulated GitLab page: wireservice/csvkit / Contributors]
Contributor             | Commits (all time, main branch)
Christopher Groskopf   | 312
James McKinney         | 94
Nikhil Sivagnanam      | 27
Celine Soyer           | 18
(All-time contributions to master branch; merge commits included.)
"""


# ---------------------------------------------------------------------------
# wa1_n3 — 3 agents, info-retrieval tasks across gitlab / map / shopping
# ---------------------------------------------------------------------------

WA1_N3 = WebArenaScenario(
    scenario_id="wa1_n3",
    description=(
        "WebArena evaluation: 3 LLM agents working on information-retrieval "
        "tasks across GitLab (task 311), OpenStreetMap (task 52), and Shopping "
        "Admin (task 279).  Tests DACS context isolation across heterogeneous "
        "web-task domains."
    ),
    agents=[
        # ------------------------------------------------------------------
        # a1 — GitLab: top contributor by commits (task_id 311)
        # Intent: "Who has made the most contributions, in terms of commits,
        #          to the pytorch-GAN project?"
        # Reference answer (exact_match): "Erik Linder-Norén"
        # ------------------------------------------------------------------
        WebArenaAgentSpec(
            agent_id="a1",
            webarena_task_id=311,
            webarena_site="gitlab",
            task_description=(
                "Analyse the contributor data for the pytorch-GAN GitLab project "
                "and report who has made the most commits."
            ),
            simulated_page_context=_PAGE_GITLAB_311,
            decision_hints=(
                "1. Branch scope: whether to count commits across all branches "
                "or restrict to the default (main) branch only when determining "
                "the top contributor.\n"
                "2. Merge commit inclusion: whether merge commits should be counted "
                "alongside regular commits when comparing contribution totals.\n"
                "3. Tie-breaking: what criterion to use if two contributors have "
                "the same commit count."
            ),
            rubrics=[
                DecisionRubric(
                    topic="branch_scope",
                    correct_keywords=["main", "default branch", "main branch", "master"],
                    judge_context=(
                        "WebArena's contributor ranking counts commits on the default "
                        "(main) branch only.  A correct response recommends using the "
                        "default or main branch as the scope for counting commits."
                    ),
                    webarena_eval="must_include",
                    reference_answer="Erik Linder-Norén",
                ),
                DecisionRubric(
                    topic="merge_commit_inclusion",
                    correct_keywords=["include", "count", "merge commit", "merge"],
                    judge_context=(
                        "Merge commits are included in GitLab's default contributor "
                        "count.  A correct response recommends including merge commits "
                        "rather than filtering them out."
                    ),
                    webarena_eval="must_include",
                    reference_answer="Erik Linder-Norén",
                ),
                DecisionRubric(
                    topic="final_answer",
                    correct_keywords=["erik", "linder", "norén", "lindenörn", "noren"],
                    judge_context=(
                        "The correct answer is Erik Linder-Norén.  A correct response "
                        "names Erik Linder-Norén (or a close spelling) as the top "
                        "contributor to the pytorch-GAN project."
                    ),
                    webarena_eval="exact_match",
                    reference_answer="Erik Linder-Norén",
                ),
            ],
        ),

        # ------------------------------------------------------------------
        # a2 — OpenStreetMap: walking time CMU → Starbucks Craig St (task 52)
        # Intent: "How long does it take to walk from CMU to Starbucks on Craig Street?"
        # Reference answer (fuzzy_match): "7 min"
        # ------------------------------------------------------------------
        WebArenaAgentSpec(
            agent_id="a2",
            webarena_task_id=52,
            webarena_site="map",
            task_description=(
                "Calculate the walking time from Carnegie Mellon University to "
                "the Starbucks on Craig Street, Pittsburgh."
            ),
            simulated_page_context=_PAGE_MAP_52,
            decision_hints=(
                "1. Walking pace: whether to use the standard average walking "
                "speed (~4 km/h) or a faster pace for the time estimate.\n"
                "2. Route selection: whether to report the shortest-distance "
                "route or the fastest-walking route when they differ.\n"
                "3. Rounding: whether to round the time estimate to the nearest "
                "minute or report exact seconds."
            ),
            rubrics=[
                DecisionRubric(
                    topic="walking_pace",
                    correct_keywords=["average", "standard", "4 km", "normal", "default"],
                    judge_context=(
                        "OpenStreetMap uses a standard average walking pace (~4 km/h) "
                        "for its routing estimates.  A correct response recommends "
                        "using the standard/average pace rather than a custom faster "
                        "or slower speed."
                    ),
                    webarena_eval="fuzzy_match",
                    reference_answer="7 min",
                ),
                DecisionRubric(
                    topic="route_type",
                    correct_keywords=["fastest", "shortest", "walking route", "pedestrian"],
                    judge_context=(
                        "The standard routing for walking tasks on OpenStreetMap uses "
                        "the shortest walking route.  A correct response selects the "
                        "walking/pedestrian routing rather than a driving or cycling route."
                    ),
                    webarena_eval="fuzzy_match",
                    reference_answer="7 min",
                ),
                DecisionRubric(
                    topic="final_answer",
                    correct_keywords=["7 min", "7 minute", "seven minute", "7-min"],
                    judge_context=(
                        "The correct answer is 7 minutes.  A correct response reports "
                        "approximately 7 minutes as the walking time from CMU to the "
                        "Starbucks on Craig Street."
                    ),
                    webarena_eval="fuzzy_match",
                    reference_answer="7 min",
                ),
            ],
        ),

        # ------------------------------------------------------------------
        # a3 — Shopping Admin: top-selling product type Q1 2022 (task 279)
        # Intent: "What is the top-1 best-selling product type in Quarter 1 2022?"
        # Reference answer (exact_match): "Yoga ball"
        # ------------------------------------------------------------------
        WebArenaAgentSpec(
            agent_id="a3",
            webarena_task_id=279,
            webarena_site="shopping_admin",
            task_description=(
                "Determine the top-1 best-selling product type at the store "
                "for Quarter 1 (January–March) of 2022."
            ),
            simulated_page_context=_PAGE_SHOPPING_279,
            decision_hints=(
                "1. Ranking metric: whether to rank product types by units sold "
                "or by revenue generated when identifying 'best-selling'.\n"
                "2. Granularity: whether 'product type' means a broad category "
                "(e.g. sports equipment) or a specific SKU-level product name.\n"
                "3. Quarter definition: whether Quarter 1 covers January–March "
                "or some other date range in the store's fiscal calendar."
            ),
            rubrics=[
                DecisionRubric(
                    topic="ranking_metric",
                    correct_keywords=["units", "units sold", "quantity", "volume", "count"],
                    judge_context=(
                        "WebArena's shopping-admin tasks define 'best-selling' by units "
                        "sold (not revenue).  A correct response recommends ranking by "
                        "quantity/units sold rather than by revenue."
                    ),
                    webarena_eval="exact_match",
                    reference_answer="Yoga ball",
                ),
                DecisionRubric(
                    topic="product_granularity",
                    correct_keywords=["product type", "type", "category", "specific product"],
                    judge_context=(
                        "'Product type' in this context refers to the specific product "
                        "name (e.g. 'Yoga ball'), not a broad category like 'fitness "
                        "equipment'.  A correct response identifies the specific product "
                        "type name rather than a categorical grouping."
                    ),
                    webarena_eval="exact_match",
                    reference_answer="Yoga ball",
                ),
                DecisionRubric(
                    topic="final_answer",
                    correct_keywords=["yoga ball", "yoga", "yoga-ball"],
                    judge_context=(
                        "The correct answer is 'Yoga ball'.  A correct response names "
                        "Yoga ball as the top-selling product type in Q1 2022."
                    ),
                    webarena_eval="exact_match",
                    reference_answer="Yoga ball",
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# wa2_n5 — 5 agents, adds reddit + gitlab second task
# ---------------------------------------------------------------------------

WA2_N5 = WebArenaScenario(
    scenario_id="wa2_n5",
    description=(
        "WebArena evaluation: 5 LLM agents across gitlab (311, 312), "
        "map (52), shopping_admin (279), and reddit (67).  Tests DACS "
        "context isolation at N=5 with heterogeneous web domains."
    ),
    agents=[
        # a1–a3: identical to wa1_n3
        WA1_N3.agents[0],
        WA1_N3.agents[1],
        WA1_N3.agents[2],

        # ------------------------------------------------------------------
        # a4 — Reddit: single-book recommendation posts (task_id 67)
        # Intent: "Among the top 10 posts in 'books' forum, show me the book
        #          names from posts that recommend a single book."
        # Reference answer (must_include): ["A Christmas Carol", "The Hobbit"]
        # ------------------------------------------------------------------
        WebArenaAgentSpec(
            agent_id="a4",
            webarena_task_id=67,
            webarena_site="reddit",
            task_description=(
                "From the top 10 posts in the r/books forum, identify which "
                "posts recommend a single book and list those book names."
            ),
            simulated_page_context=_PAGE_REDDIT_67,
            decision_hints=(
                "1. Single-book criterion: whether a post 'recommends a single "
                "book' only when the entire post is dedicated to that one book, "
                "or also when one book is clearly highlighted among others.\n"
                "2. Ranking method: whether 'top 10' refers to the posts ranked "
                "by upvotes/score or by recency (newest posts).\n"
                "3. Title vs body: whether to assess the post's title only or "
                "also its body text when classifying as single-book recommendation."
            ),
            rubrics=[
                DecisionRubric(
                    topic="single_book_criterion",
                    correct_keywords=["dedicated", "single", "only one", "exclusively", "entire post"],
                    judge_context=(
                        "A post 'recommends a single book' when the post is exclusively "
                        "about one book — not a list or comparison.  A correct response "
                        "applies a strict criterion: the post is entirely focused on "
                        "recommending or discussing one specific book title."
                    ),
                    webarena_eval="must_include",
                    reference_answer="A Christmas Carol, The Hobbit",
                ),
                DecisionRubric(
                    topic="ranking_method",
                    correct_keywords=["upvote", "score", "votes", "top", "ranked by score"],
                    judge_context=(
                        "WebArena's 'top N posts' uses Reddit's score (upvotes) ranking, "
                        "not recency.  A correct response confirms using upvote/score "
                        "ranking to select the top 10 posts."
                    ),
                    webarena_eval="must_include",
                    reference_answer="A Christmas Carol, The Hobbit",
                ),
                DecisionRubric(
                    topic="final_answer",
                    correct_keywords=[
                        "a christmas carol", "christmas carol",
                        "the hobbit", "hobbit",
                    ],
                    judge_context=(
                        "The correct answer includes 'A Christmas Carol' and 'The Hobbit'. "
                        "A correct response names both of these book titles."
                    ),
                    webarena_eval="must_include",
                    reference_answer="A Christmas Carol, The Hobbit",
                ),
            ],
        ),

        # ------------------------------------------------------------------
        # a5 — GitLab: top contributor to csvkit (task_id 312)
        # Intent: "Who has made the most contributions, in terms of commits,
        #          to the csvkit project?"
        # Reference answer (exact_match): "Christopher Groskopf"
        # ------------------------------------------------------------------
        WebArenaAgentSpec(
            agent_id="a5",
            webarena_task_id=312,
            webarena_site="gitlab",
            task_description=(
                "Find who has made the most commits to the csvkit project "
                "on GitLab (wireservice/csvkit)."
            ),
            simulated_page_context=_PAGE_GITLAB_312,
            decision_hints=(
                "1. Branch scope: whether to count commits on all branches or "
                "only the main/master branch for determining the top contributor.\n"
                "2. Time period: whether to count all-time contributions or only "
                "recent contributions (e.g., last 12 months).\n"
                "3. Attribution: whether co-authored commits should count toward "
                "only the primary committer or be split across all authors."
            ),
            rubrics=[
                DecisionRubric(
                    topic="branch_scope",
                    correct_keywords=["main", "master", "default", "default branch"],
                    judge_context=(
                        "GitLab's contributor ranking counts commits on the default "
                        "(master/main) branch.  A correct response recommends using "
                        "the default/master branch as the scope."
                    ),
                    webarena_eval="exact_match",
                    reference_answer="Christopher Groskopf",
                ),
                DecisionRubric(
                    topic="time_period",
                    correct_keywords=["all-time", "all time", "total", "entire history", "all"],
                    judge_context=(
                        "WebArena's contributor tasks count all-time commits, not "
                        "just recent ones.  A correct response recommends counting "
                        "all-time (total historical) contributions."
                    ),
                    webarena_eval="exact_match",
                    reference_answer="Christopher Groskopf",
                ),
                DecisionRubric(
                    topic="final_answer",
                    correct_keywords=["christopher groskopf", "groskopf", "christopher"],
                    judge_context=(
                        "The correct answer is Christopher Groskopf.  A correct response "
                        "names Christopher Groskopf as the top contributor to csvkit."
                    ),
                    webarena_eval="exact_match",
                    reference_answer="Christopher Groskopf",
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

WEB_SCENARIOS: dict[str, WebArenaScenario] = {
    "wa1_n3": WA1_N3,
    "wa2_n5": WA2_N5,
}
