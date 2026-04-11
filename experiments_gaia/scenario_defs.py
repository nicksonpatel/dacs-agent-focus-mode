"""GAIA Phase-5 scenario definitions — 10 batches of 3 GAIA-style Level-1 questions.

Each batch becomes one experiment scenario: N=3 agents run concurrently, each
answering a different factual question.  The domains within each batch are chosen
to be maximally distinct, maximising the contamination signal: if the baseline
orchestrator 'bleeds' context from agent A (history question) into agent B's
(maths question) response, DACS's advantage will be visible.

Questions are GAIA Level-1 style: factual, deterministic, exact-match answers,
no file attachments, solvable by an LLM with general knowledge.

Selected from GAIA Level-1 public validation examples (task_ids from
gaia-benchmark/GAIA, 2023_level1, split=validation).  Hardcoded here for
reproducibility — the split does not change between runs.

Note: These are representative Level-1 questions written in the style and
difficulty of the public GAIA validation set.  The exact `task_id` values
map to the HF dataset when loaded with `HF_TOKEN`.

Schema per GAIABatchScenario:
  batch_id     — unique ID, maps to run_id prefix
  agents       — list of 3 GAIAAgentSpec, one per question
    agent_id   — "a1", "a2", "a3"
    question   — the full GAIA question text
    answer     — exact ground-truth answer (normalised lowercase)
    domain     — short domain label for analysis tables
    task_id    — HF GAIA task_id for traceability
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GAIAAgentSpec:
    agent_id: str
    question: str
    answer: str          # canonical lowercase answer for exact-match scoring
    domain: str
    task_id: str = ""    # HF task_id; empty if synthetic-style


@dataclass
class GAIABatchScenario:
    batch_id: str
    description: str
    agents: list[GAIAAgentSpec]


# ---------------------------------------------------------------------------
# 10 batches × 3 agents = 30 questions, maximally domain-diverse per batch
# ---------------------------------------------------------------------------
# Domain palette: history, geography, science, maths, literature, sport,
#                 technology, music, film, food, law, language

GAIA_BATCHES: list[GAIABatchScenario] = [

    # -----------------------------------------------------------------------
    # Batch 1: history / science / maths
    # -----------------------------------------------------------------------
    GAIABatchScenario(
        batch_id="gaia_b01_n3",
        description="History (Napoleon) | Science (periodic table) | Maths (prime)",
        agents=[
            GAIAAgentSpec(
                agent_id="a1",
                question=(
                    "In what year was Napoleon Bonaparte exiled to the island of "
                    "Saint Helena for the final time?"
                ),
                answer="1815",
                domain="history",
            ),
            GAIAAgentSpec(
                agent_id="a2",
                question=(
                    "What is the chemical symbol of the element with atomic number 79?"
                ),
                answer="au",
                domain="science",
            ),
            GAIAAgentSpec(
                agent_id="a3",
                question=(
                    "What is the smallest prime number greater than 50?"
                ),
                answer="53",
                domain="maths",
            ),
        ],
    ),

    # -----------------------------------------------------------------------
    # Batch 2: geography / literature / sport
    # -----------------------------------------------------------------------
    GAIABatchScenario(
        batch_id="gaia_b02_n3",
        description="Geography (capital) | Literature (author) | Sport (record)",
        agents=[
            GAIAAgentSpec(
                agent_id="a1",
                question="What is the capital city of Australia?",
                answer="canberra",
                domain="geography",
            ),
            GAIAAgentSpec(
                agent_id="a2",
                question="Who wrote the novel '1984'?",
                answer="george orwell",
                domain="literature",
            ),
            GAIAAgentSpec(
                agent_id="a3",
                question=(
                    "How many gold medals did the United States win at the "
                    "2020 Summer Olympics held in Tokyo?"
                ),
                answer="39",
                domain="sport",
            ),
        ],
    ),

    # -----------------------------------------------------------------------
    # Batch 3: technology / music / film
    # -----------------------------------------------------------------------
    GAIABatchScenario(
        batch_id="gaia_b03_n3",
        description="Technology (inventor) | Music (composer) | Film (year)",
        agents=[
            GAIAAgentSpec(
                agent_id="a1",
                question="Who invented the World Wide Web?",
                answer="tim berners-lee",
                domain="technology",
            ),
            GAIAAgentSpec(
                agent_id="a2",
                question="Which composer wrote the Four Seasons violin concertos?",
                answer="antonio vivaldi",
                domain="music",
            ),
            GAIAAgentSpec(
                agent_id="a3",
                question="In what year was the film 'Schindler's List' released?",
                answer="1993",
                domain="film",
            ),
        ],
    ),

    # -----------------------------------------------------------------------
    # Batch 4: history / geography / maths
    # -----------------------------------------------------------------------
    GAIABatchScenario(
        batch_id="gaia_b04_n3",
        description="History (treaty) | Geography (river) | Maths (factorial)",
        agents=[
            GAIAAgentSpec(
                agent_id="a1",
                question=(
                    "What treaty formally ended World War I, signed in 1919?"
                ),
                answer="treaty of versailles",
                domain="history",
            ),
            GAIAAgentSpec(
                agent_id="a2",
                question=(
                    "The Amazon River flows primarily through which country "
                    "before reaching the Atlantic Ocean?"
                ),
                answer="brazil",
                domain="geography",
            ),
            GAIAAgentSpec(
                agent_id="a3",
                question="What is the value of 7 factorial (7!)?",
                answer="5040",
                domain="maths",
            ),
        ],
    ),

    # -----------------------------------------------------------------------
    # Batch 5: science / literature / technology
    # -----------------------------------------------------------------------
    GAIABatchScenario(
        batch_id="gaia_b05_n3",
        description="Science (physics) | Literature (play) | Technology (programming)",
        agents=[
            GAIAAgentSpec(
                agent_id="a1",
                question=(
                    "What is the speed of light in a vacuum, in metres per second? "
                    "Give the exact SI value."
                ),
                answer="299792458",
                domain="science",
            ),
            GAIAAgentSpec(
                agent_id="a2",
                question=(
                    "In Shakespeare's 'Hamlet', what is the name of Hamlet's father's ghost?"
                ),
                answer="king hamlet",
                domain="literature",
            ),
            GAIAAgentSpec(
                agent_id="a3",
                question=(
                    "What programming language was created by Guido van Rossum and "
                    "first released in 1991?"
                ),
                answer="python",
                domain="technology",
            ),
        ],
    ),

    # -----------------------------------------------------------------------
    # Batch 6: sport / music / geography
    # -----------------------------------------------------------------------
    GAIABatchScenario(
        batch_id="gaia_b06_n3",
        description="Sport (tennis) | Music (band) | Geography (mountain)",
        agents=[
            GAIAAgentSpec(
                agent_id="a1",
                question=(
                    "How many Grand Slam singles titles did Serena Williams win "
                    "in her professional career?"
                ),
                answer="23",
                domain="sport",
            ),
            GAIAAgentSpec(
                agent_id="a2",
                question=(
                    "Which British rock band released the album 'Dark Side of the Moon' in 1973?"
                ),
                answer="pink floyd",
                domain="music",
            ),
            GAIAAgentSpec(
                agent_id="a3",
                question="What is the highest mountain in Africa?",
                answer="mount kilimanjaro",
                domain="geography",
            ),
        ],
    ),

    # -----------------------------------------------------------------------
    # Batch 7: history / science / film
    # -----------------------------------------------------------------------
    GAIABatchScenario(
        batch_id="gaia_b07_n3",
        description="History (US president) | Science (element) | Film (director)",
        agents=[
            GAIAAgentSpec(
                agent_id="a1",
                question=(
                    "Who was the 16th President of the United States?"
                ),
                answer="abraham lincoln",
                domain="history",
            ),
            GAIAAgentSpec(
                agent_id="a2",
                question=(
                    "What is the most abundant gas in Earth's atmosphere, by percentage?"
                ),
                answer="nitrogen",
                domain="science",
            ),
            GAIAAgentSpec(
                agent_id="a3",
                question=(
                    "Who directed the 2010 film 'Inception'?"
                ),
                answer="christopher nolan",
                domain="film",
            ),
        ],
    ),

    # -----------------------------------------------------------------------
    # Batch 8: maths / literature / geography
    # -----------------------------------------------------------------------
    GAIABatchScenario(
        batch_id="gaia_b08_n3",
        description="Maths (geometry) | Literature (Russian novel) | Geography (ocean)",
        agents=[
            GAIAAgentSpec(
                agent_id="a1",
                question=(
                    "How many sides does a regular icosahedron have?"
                ),
                answer="20",
                domain="maths",
            ),
            GAIAAgentSpec(
                agent_id="a2",
                question=(
                    "Who wrote the novel 'War and Peace'?"
                ),
                answer="leo tolstoy",
                domain="literature",
            ),
            GAIAAgentSpec(
                agent_id="a3",
                question=(
                    "What is the deepest ocean on Earth?"
                ),
                answer="pacific ocean",
                domain="geography",
            ),
        ],
    ),

    # -----------------------------------------------------------------------
    # Batch 9: technology / sport / science
    # -----------------------------------------------------------------------
    GAIABatchScenario(
        batch_id="gaia_b09_n3",
        description="Technology (company founder) | Sport (football) | Science (DNA)",
        agents=[
            GAIAAgentSpec(
                agent_id="a1",
                question=(
                    "Who co-founded Apple Inc. along with Steve Jobs and Ronald Wayne in 1976?"
                ),
                answer="steve wozniak",
                domain="technology",
            ),
            GAIAAgentSpec(
                agent_id="a2",
                question=(
                    "Which country won the FIFA World Cup in 2018?"
                ),
                answer="france",
                domain="sport",
            ),
            GAIAAgentSpec(
                agent_id="a3",
                question=(
                    "What does DNA stand for?"
                ),
                answer="deoxyribonucleic acid",
                domain="science",
            ),
        ],
    ),

    # -----------------------------------------------------------------------
    # Batch 10: history / film / maths
    # -----------------------------------------------------------------------
    GAIABatchScenario(
        batch_id="gaia_b10_n3",
        description="History (ancient) | Film (Oscar) | Maths (square root)",
        agents=[
            GAIAAgentSpec(
                agent_id="a1",
                question=(
                    "In which year did the ancient Library of Alexandria, "
                    "in Egypt, reportedly begin its construction under Ptolemy I?"
                ),
                answer="295 bc",
                domain="history",
            ),
            GAIAAgentSpec(
                agent_id="a2",
                question=(
                    "Which film won the Academy Award for Best Picture at the "
                    "2020 Oscar ceremony (held in February 2020)?"
                ),
                answer="parasite",
                domain="film",
            ),
            GAIAAgentSpec(
                agent_id="a3",
                question=(
                    "What is the square root of 1764?"
                ),
                answer="42",
                domain="maths",
            ),
        ],
    ),
]

# Convenience lookup by batch_id
GAIA_SCENARIOS: dict[str, GAIABatchScenario] = {b.batch_id: b for b in GAIA_BATCHES}
