#!/usr/bin/env python3
"""Generate One-vs-All V1 long-chunk box-office contextual-reuse datasets.

This is a separate follow-up to the original 4-chunk box-office matrix.  It
keeps the same cache-conditioning semantics but emits Diego/LongBench-style
rows with 10 passage chunks and approximately 512 tokens per movie chunk.
Each row asks for the highest BOX_OFFICE_MUSD across every named candidate in
the context.  The top two candidates remain tracked for cache-conditioning
cells, but solving the question requires scanning all ten movie chunks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROMPT_VARIANTS: Dict[str, str] = {
    "direct": (
        "Answer the question based ONLY on the provided context. "
        "Be concise - output just the answer, no explanation."
    ),
    "careful_max": (
        "Answer based ONLY on the provided movie dossiers. "
        "Read BOX_OFFICE_MUSD as an integer for every named candidate, choose the largest value, "
        "and output exactly one FILM-ID with no explanation."
    ),
    "closed_world": (
        "Use only the synthetic movie dossiers below. Ignore outside/world knowledge. "
        "For ranking, compare only the BOX_OFFICE_MUSD integer fields across all named candidates. "
        "Return exactly one FILM-ID and no other text."
    ),
    "field_line_max": (
        "Answer with exactly one FILM-ID and nothing else. Do not explain, list values, or show work. "
        "Use only the BENCHMARK_DOSSIER blocks below. For each named candidate, use its "
        "'BOX_OFFICE_MUSD: <integer>' line and ignore all other numbers, including RELEASE_YEAR, "
        "RUNTIME_MIN, awards, and city population."
    ),
}
DEFAULT_PROMPT_VARIANT = "closed_world"
CELL_CODES = ("<<", "<>", "><", ">>")
DOSSIER_RE = re.compile(
    r"BENCHMARK_DOSSIER\n(.*?)(?=\n\nBENCHMARK_DOSSIER\n|\n\nEXTENDED_COMPARISON_FACTS\n|\n\nAcknowledge with OK\.|\Z)",
    flags=re.DOTALL,
)
FIELD_RE = re.compile(r"^([A-Z_]+):\s*(.*)$", flags=re.MULTILINE)
RENDER_STYLE = os.environ.get("ONE_VS_ALL_RELEASE_STYLE", "plain_catalog").strip() or "plain_catalog"


@dataclass(frozen=True)
class Film:
    entity_id: str
    title: str
    director: str
    release_year: int
    starlight_awards: int
    box_office_musd: int
    genre: str
    studio: str
    country: str
    runtime_min: int
    cast: str = ""
    setting_city: str = ""
    setting_city_population_mil: str = ""


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def parse_film_block(block: str) -> Optional[Film]:
    fields = {key: value.strip() for key, value in FIELD_RE.findall(block)}
    required = [
        "ENTITY_ID",
        "TITLE",
        "DIRECTOR",
        "RELEASE_YEAR",
        "STARLIGHT_AWARDS",
        "BOX_OFFICE_MUSD",
        "GENRE",
        "STUDIO",
        "COUNTRY",
        "RUNTIME_MIN",
    ]
    if any(key not in fields for key in required):
        return None
    return Film(
        entity_id=fields["ENTITY_ID"],
        title=fields["TITLE"],
        director=fields["DIRECTOR"],
        release_year=int(fields["RELEASE_YEAR"]),
        starlight_awards=int(fields["STARLIGHT_AWARDS"]),
        box_office_musd=int(fields["BOX_OFFICE_MUSD"]),
        genre=fields["GENRE"],
        studio=fields["STUDIO"],
        country=fields["COUNTRY"],
        runtime_min=int(fields["RUNTIME_MIN"]),
        cast=fields.get("CAST", ""),
        setting_city=fields.get("SETTING_CITY", ""),
        setting_city_population_mil=fields.get("SETTING_CITY_POPULATION_MIL", ""),
    )


def load_films_from_matrix(paths: Iterable[Path]) -> List[Film]:
    by_id: Dict[str, Film] = {}
    for path in paths:
        for row in load_jsonl(path):
            prompt = str(row.get("turn_1_poison_prompt") or "")
            for block in DOSSIER_RE.findall(prompt):
                film = parse_film_block(block)
                if film is not None:
                    by_id[film.entity_id] = film
    films = sorted(by_id.values(), key=lambda film: (film.box_office_musd, film.entity_id), reverse=True)
    if len(films) < 24:
        raise RuntimeError(f"Need at least 24 parsed films; got {len(films)}")
    return films


def default_source_paths() -> List[Path]:
    gen_root = Path(__file__).resolve().parents[2]
    candidates = [
        [
            gen_root / "data/phase2_boxoffice_balanced_matrix_s7.jsonl",
            gen_root / "data/phase2_boxoffice_balanced_matrix_s11.jsonl",
            gen_root / "data/supplemental_boxoffice_corpus_v1_50.jsonl",
        ],
    ]
    return next((paths for paths in candidates if all(path.exists() for path in paths)), candidates[0])


def build_token_counter(tokenizer_path: Optional[Path]):
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    except Exception as exc:
        print(f"WARNING: tokenizer unavailable at {tokenizer_path}: {exc}")
        return None

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    return count_tokens


def base_movie_chunk(film: Film) -> str:
    lines = [
        "BENCHMARK_DOSSIER",
        f"ENTITY_ID: {film.entity_id}",
        f"TITLE: {film.title}",
        f"DIRECTOR: {film.director}",
        f"RELEASE_YEAR: {film.release_year}",
        f"STARLIGHT_AWARDS: {film.starlight_awards}",
        f"BOX_OFFICE_MUSD: {film.box_office_musd}",
        f"GENRE: {film.genre}",
        f"STUDIO: {film.studio}",
        f"COUNTRY: {film.country}",
        f"RUNTIME_MIN: {film.runtime_min}",
    ]
    if film.cast:
        lines.append(f"CAST: {film.cast}")
    city = film.setting_city or synthetic_city(film)
    city_pop = film.setting_city_population_mil or synthetic_city_pop(film)
    if RENDER_STYLE == "imdb_control":
        body = [
            "",
            "CATALOG_ENTRY",
            f"Synopsis: {film.title} is a {film.genre.lower()} feature associated with {city}.",
            (
                f"Production: The record credits {film.director} as director and {film.studio} as "
                f"the studio, with {film.country} listed as the primary market territory."
            ),
            (
                f"Release and reception: The dossier records release year {film.release_year}, "
                f"runtime {film.runtime_min} minutes, Starlight Awards {film.starlight_awards}, "
                f"and box office {film.box_office_musd} million USD."
            ),
            (
                f"Cast and setting: The listed cast is {film.cast or film.title + ' ensemble'}, and "
                f"the archive setting reference is {city} with synthetic population {city_pop} million."
            ),
        ]
    elif RENDER_STYLE == "no_extra_numbers":
        body = [
            "",
            "CATALOG_ENTRY",
            f"Synopsis: {film.title} is filed as a {film.genre.lower()} feature associated with {city}.",
            (
                f"Production: The record credits {film.director} as director and {film.studio} as "
                f"the studio, with {film.country} listed as the main territory."
            ),
            f"Catalog note: The listing groups cast, setting, studio, genre, and release context into a compact dossier for {film.title}.",
            f"Setting: The archive location reference for the title is {city}.",
        ]
    elif RENDER_STYLE == "boxoffice_salient":
        body = [
            "",
            "CATALOG_ENTRY",
            f"Synopsis: {film.title} is filed as a {film.genre.lower()} feature associated with {city}.",
            (
                f"Production: The record credits {film.director} as director and {film.studio} as "
                f"the studio, with {film.country} listed as the main territory."
            ),
            f"Commercial summary: The recorded box office for {film.title} is {film.box_office_musd} million USD.",
            f"Catalog note: The listing groups cast, setting, studio, genre, and release context into a compact dossier for {film.title}.",
        ]
    elif RENDER_STYLE == "plain_catalog":
        body = [
            "",
            "CATALOG_ENTRY",
            (
                f"Overview: {film.title} is a {film.release_year} {film.genre.lower()} feature directed by "
                f"{film.director} for {film.studio}. In short directory listings, the film is usually identified "
                f"through its {film.country} market line and its association with {city}."
            ),
            (
                f"Cast and setting: The credited cast is {film.cast or film.title + ' ensemble'}. "
                f"Reference copy keeps {city} attached to the title as its standing location marker rather than "
                f"expanding into a long plot synopsis."
            ),
            (
                f"Release profile: The entry presents the film as a full-length commercial feature with a runtime "
                f"of {film.runtime_min} minutes. The dossier keeps the release identity tied to studio, market, "
                f"cast, and setting in the style of a compact film guide."
            ),
            (
                f"Commercial note: The recorded box office for {film.title} is {film.box_office_musd} million USD, "
                f"and the listing treats that figure as part of the title's market profile rather than as a review note."
            ),
        ]
    else:
        raise ValueError(f"Unknown ONE_VS_ALL_RELEASE_STYLE={RENDER_STYLE}")
    lines.extend(
        body
        + [
            "",
            "RELEASE_AND_REFERENCE_NOTES",
            f"SETTING_CITY: {city}",
            f"SETTING_CITY_POPULATION_MIL: {city_pop}",
        ]
    )
    return "\n".join(lines)


def stable_int(text: str) -> int:
    value = 0
    for ch in text:
        value = ((value * 131) + ord(ch)) % (2**32)
    return value


def synthetic_city(film: Film) -> str:
    cities = [
        "Aurelia Prime",
        "Vesper Gate",
        "Northreach City",
        "Solara Bay",
        "Cindermark",
        "Kestrel Port",
        "Lunaris",
        "Red Harbor",
        "Novastad",
        "Helion Cross",
    ]
    return cities[stable_int(film.entity_id) % len(cities)]


def synthetic_city_pop(film: Film) -> str:
    return f"{3.2 + (stable_int(film.title) % 110) / 10.0:.1f}"


def plain_catalog_tail_sentences(film: Film) -> List[str]:
    city = film.setting_city or synthetic_city(film)
    cast_text = film.cast or f"{film.title} ensemble"
    key = stable_int(film.entity_id)
    release_style = ["mainstream commercial", "wide theatrical", "general release", "studio-backed", "broad release"][
        key % 5
    ]
    filing_style = ["film-guide", "directory", "yearbook", "almanac", "distribution-reference"][(key // 7) % 5]
    billing_style = ["director-led", "studio-led", "title-led", "market-led", "cast-led"][(key // 13) % 5]
    placement_style = ["contemporary thrillers", "recent commercial releases", "modern studio entries", "wide-market titles", "current genre features"][(key // 17) % 5]
    exhibition_style = ["ordinary theatrical booking", "broad commercial circulation", "standard multiplex play", "general commercial scheduling", "regular market release"][(key // 19) % 5]
    return [
        f"Studio filing: {film.studio} remains the credited company throughout the record.",
        f"Market filing: {film.country} is the territory used for release and indexing reference.",
        f"Cast filing: {cast_text} remains the recurring cast line attached to the title.",
        f"Setting filing: {city} is the place-name consistently used to anchor the film in short reference prose.",
        f"Directory note: The title is grouped with {placement_style} rather than with repertory or specialty listings.",
        f"Exhibition note: The entry reads like a film associated with {exhibition_style}.",
        f"Reference note: The record keeps to production identity, market placement, cast, setting, runtime, and gross without adding review language.",
        f"Index note: {film.title} is filed in a {billing_style} pattern that makes the title easy to locate in long title lists.",
        f"Program note: The note block favors release and reference detail over plot description.",
        f"Library note: The overall effect is closer to a compact {filing_style} record than to a pressbook or capsule review.",
        f"Release note: The title is presented as a {release_style} {film.genre.lower()} feature rather than as a limited event item.",
        f"Record line: The runtime and release year remain part of the short production summary for the film.",
        f"Reference line: Genre, studio, cast, and setting provide the main points of identification.",
        f"Distribution note: The dossier groups studio, market, and commercial performance as standard release metadata.",
        f"Listing note: The entry keeps a restrained factual tone from the main facts block through the supplemental notes.",
        f"Market line: The commercial profile remains tied to the recorded {film.box_office_musd} million USD theatrical total.",
        f"Cast line: The credited cast is repeated in plain language so the title stays easy to identify across long runs of records.",
        f"Setting line: The file continues to use {city} as the standing location reference for the film.",
        f"Archive line: The title is treated as a stable commercial record rather than as an editorial or promotional item.",
        f"Guide line: The dossier reads like a concise lookup entry prepared for practical film-reference use.",
    ]


def filler_sentences(film: Film) -> List[str]:
    key = stable_int(film.entity_id)
    campaign = ["weekday", "holiday", "festival", "late-summer", "winter", "awards-season", "back-to-school"][key % 7]
    release = [
        "domestic-first",
        "regional-roadshow",
        "platform",
        "wide",
        "limited",
        "festival-to-wide",
        "preview-first",
    ][key % 7]
    audience = [
        "family",
        "genre",
        "premium-format",
        "matinee",
        "subscriber",
        "date-night",
        "rewatch",
    ][key % 7]
    posture = ["steady", "front-loaded", "leggy", "holiday-skewing", "weekend-led", "word-of-mouth", "premium-heavy"][
        (key // 7) % 7
    ]
    archive_tag = ["regional", "national", "catalog", "festival", "late-run", "showcase", "revival"][(key // 11) % 7]
    press_tone = ["trade", "programmer", "exhibitor", "catalog", "audience", "archive", "syndicated"][(key // 13) % 7]
    city = film.setting_city or synthetic_city(film)
    city_pop = film.setting_city_population_mil or synthetic_city_pop(film)
    cast_text = film.cast or f"{film.title} ensemble"
    if RENDER_STYLE == "imdb_control":
        lines = [
            (
                f"Overview: {film.title} is listed as a {film.genre.lower()} title from {film.studio}, "
                f"directed by {film.director} and released through the {film.country} market."
            ),
            (
                f"Plot note: Archive copy places the story around {city}, giving the listing a consistent "
                "setting reference alongside the core production details."
            ),
            (
                f"Release profile: Trade summaries describe a {release} launch supported by a {campaign} "
                f"campaign, with the reported theatrical gross recorded at {film.box_office_musd} million USD."
            ),
            (
                f"Cast note: The principal cast is listed as {cast_text}, and the running time is recorded as "
                f"{film.runtime_min} minutes."
            ),
            (
                f"Catalog note: The studio-credit and territory pairing of {film.studio} and {film.country} "
                f"matches the {archive_tag} filing style used for the surrounding entries."
            ),
            (
                f"Reception note: Coverage describes the run as {posture} and often links the title to a "
                f"{audience} audience segment."
            ),
            (
                f"Programming note: Pressbook and exhibitor copy follow a {press_tone} tone, emphasizing "
                f"genre, cast, and release framing for {film.title}."
            ),
            (
                f"Setting note: The archive listing pairs the film with {city} and a population marker of "
                f"{city_pop} million to keep the locale description concrete."
            ),
            (
                f"Awards note: The title is recorded with {film.starlight_awards} Starlight Awards, a figure "
                "typically cited in catalog sidebars and year-end listings."
            ),
            (
                f"Runtime note: At {film.runtime_min} minutes, the film is usually filed as a full-length "
                f"{film.genre.lower()} feature rather than a specialty short or anthology segment."
            ),
            (
                f"Market brief: Trade coverage typically highlights the {film.genre.lower()} positioning, "
                f"{campaign} rollout timing, and the box-office result in the same summary paragraph."
            ),
            (
                f"Release note: Listings usually place {film.title} in the {film.release_year} release calendar "
                f"with {film.studio} handling distribution and venue booking."
            ),
            (
                f"Cast and crew note: Directory entries pair {film.director} with {cast_text} so readers can "
                "identify the title quickly in long-form film indexes."
            ),
            (
                f"Exhibitor note: Venue-facing summaries often describe the run as {release} and mention a "
                f"{campaign} push when discussing attendance patterns."
            ),
            (
                f"Archive note: The entry keeps together the year, studio, territory, setting, and gross so it "
                "reads like a compact catalog card rather than a bare spreadsheet export."
            ),
        ]
    elif RENDER_STYLE == "no_extra_numbers":
        lines = [
            f"Overview: {film.title} is listed as a {film.genre.lower()} title from {film.studio}.",
            f"Synopsis note: Catalog copy links the film to {city} and treats that setting as a stable archive reference.",
            f"Production note: The listing keeps {film.director} tied to {film.studio} so the record reads like a compact catalog entry.",
            f"Market note: The entry places the film in the {film.country} market and describes the release in plain trade language.",
            f"Cast note: Directory-style summaries pair the title with {cast_text} to make the listing easy to scan.",
            f"Program note: Archive copy emphasizes genre, cast, studio, and setting rather than long narrative detail.",
            f"Catalog note: The record is written to resemble a short film index entry rather than a spreadsheet dump.",
            f"Exhibitor note: Venue-facing summaries usually describe the release plainly and keep the tone factual.",
            f"Archive note: The entry keeps together title, studio, territory, cast, and setting in a consistent order.",
            f"Genre note: The {film.genre.lower()} label is repeated in catalog prose so the title reads like a familiar listing.",
            f"Style note: The prose stays short and uniform so the dossier remains easy to scan in long prompts.",
            f"Index note: Reference copy for {film.title} follows the same filing style used across the surrounding entries.",
            f"Setting note: {city} remains the location anchor for the record and is repeated to stabilize the listing.",
            f"Credit note: The film remains associated with {film.director} and {film.studio} throughout the entry.",
            f"Listing note: The dossier keeps descriptive prose separate from the structured numeric fields above.",
        ]
    elif RENDER_STYLE == "boxoffice_salient":
        lines = [
            f"Overview: {film.title} is listed as a {film.genre.lower()} title from {film.studio}.",
            f"Synopsis note: Catalog copy links the film to {city} and treats that setting as a stable archive reference.",
            f"Production note: The listing keeps {film.director} tied to {film.studio} so the record reads like a compact catalog entry.",
            f"Commercial note: Trade summaries for the title keep the recorded box-office total in view when describing its run.",
            f"Cast note: Directory-style summaries pair the title with {cast_text} to make the listing easy to scan.",
            f"Program note: Archive copy emphasizes genre, cast, studio, setting, and commercial performance in plain language.",
            f"Catalog note: The record is written to resemble a short film index entry rather than a spreadsheet dump.",
            f"Exhibitor note: Venue-facing summaries usually describe the release plainly and keep the tone factual.",
            f"Archive note: The entry keeps together title, studio, territory, cast, setting, and gross in a consistent order.",
            f"Genre note: The {film.genre.lower()} label is repeated in catalog prose so the title reads like a familiar listing.",
            f"Style note: The prose stays short and uniform so the dossier remains easy to scan in long prompts.",
            f"Index note: Reference copy for {film.title} follows the same filing style used across the surrounding entries.",
            f"Setting note: {city} remains the location anchor for the record and is repeated to stabilize the listing.",
            f"Credit note: The film remains associated with {film.director} and {film.studio} throughout the entry.",
            f"Listing note: The dossier keeps descriptive prose separate from the structured numeric fields above.",
        ]
    elif RENDER_STYLE == "plain_catalog":
        lines = plain_catalog_tail_sentences(film)
    else:
        raise ValueError(f"Unknown ONE_VS_ALL_RELEASE_STYLE={RENDER_STYLE}")
    assert len(lines) == len(set(lines))
    return lines


def long_movie_chunk(film: Film, target_tokens: int, count_tokens) -> Tuple[str, Optional[int]]:
    chunk = base_movie_chunk(film)
    sentences = filler_sentences(film)
    tail_sentences = plain_catalog_tail_sentences(film) if RENDER_STYLE == "plain_catalog" else []
    if count_tokens is None:
        idx = 0
        while len(chunk.split()) < int(target_tokens * 0.72):
            if idx < len(sentences):
                line = sentences[idx]
            elif tail_sentences:
                line = tail_sentences[(idx - len(sentences)) % len(tail_sentences)]
            else:
                line = (
                    f"APPENDIX_NOTE_{idx:02d}: Supplemental synthetic archive wording preserves the long-chunk "
                    f"format for {film.entity_id} without changing any authoritative field."
                )
            chunk += "\n" + line
            idx += 1
        return chunk, None

    idx = 0
    while count_tokens(chunk) < target_tokens:
        if idx < len(sentences):
            line = sentences[idx]
        elif tail_sentences:
            line = tail_sentences[(idx - len(sentences)) % len(tail_sentences)]
        else:
            line = (
                f"Catalog appendix {idx:02d}: This supplemental benchmark prose keeps the dossier long enough "
                f"for cache-reuse testing while preserving {film.entity_id} as a stable movie record and "
                "leaving BOX_OFFICE_MUSD as the only authoritative ranking field."
            )
        chunk += "\n" + line
        idx += 1
    return chunk, count_tokens(chunk)


def status_from_predecessors(target: Film, predecessors: Sequence[Film]) -> Optional[str]:
    if not predecessors:
        return None
    bigger = sum(1 for film in predecessors if film.box_office_musd > target.box_office_musd)
    smaller = sum(1 for film in predecessors if film.box_office_musd < target.box_office_musd)
    if bigger == len(predecessors):
        return ">"
    if smaller == len(predecessors):
        return "<"
    return None


def relation_to_target(predecessor_value: Optional[int], target_value: int) -> Optional[str]:
    if predecessor_value is None:
        return None
    if predecessor_value > target_value:
        return ">"
    if predecessor_value < target_value:
        return "<"
    return "="


def condition_target(
    *,
    target: Film,
    desired_status: str,
    seen: Sequence[Film],
    rng: random.Random,
    num_chunks: int,
    conditioning_predecessors: int,
) -> Optional[List[Film]]:
    target_value = target.box_office_musd
    if desired_status == "<":
        primary = [film for film in seen if film.box_office_musd < target_value and film.entity_id != target.entity_id]
    else:
        primary = [film for film in seen if film.box_office_musd > target_value and film.entity_id != target.entity_id]
    predecessor_count = max(1, min(int(conditioning_predecessors), num_chunks - 1))
    if len(primary) < predecessor_count:
        return None
    predecessors = rng.sample(primary, k=predecessor_count)
    filler_pool = [film for film in seen if film.entity_id not in {target.entity_id, *(x.entity_id for x in predecessors)}]
    if len(filler_pool) < num_chunks - predecessor_count - 1:
        return None
    followers = rng.sample(filler_pool, k=num_chunks - predecessor_count - 1)
    context = predecessors + [target] + followers
    if status_from_predecessors(target, predecessors) != desired_status:
        return None
    return context


def stratified_candidates(
    candidates: Sequence[Film],
    *,
    rng: random.Random,
    reverse: bool,
    window_size: int,
    bins: int = 5,
) -> List[Film]:
    ordered = sorted(candidates, key=lambda film: (film.box_office_musd, film.entity_id), reverse=reverse)
    if len(ordered) <= window_size:
        rng.shuffle(ordered)
        return ordered
    chosen: List[Film] = []
    used = set()
    per_bin = max(1, math.ceil(window_size / bins))
    for bin_idx in range(bins):
        start = int(round(bin_idx * len(ordered) / bins))
        end = int(round((bin_idx + 1) * len(ordered) / bins))
        bucket = ordered[start:end]
        rng.shuffle(bucket)
        for film in bucket[:per_bin]:
            if film.entity_id not in used:
                chosen.append(film)
                used.add(film.entity_id)
            if len(chosen) >= window_size:
                break
        if len(chosen) >= window_size:
            break
    remainder = [film for film in ordered if film.entity_id not in used]
    rng.shuffle(chosen)
    rng.shuffle(remainder)
    return chosen + remainder


def instruction_for_variant(prompt_variant: str) -> str:
    return PROMPT_VARIANTS[prompt_variant]


def make_question(context: Sequence[Film], prompt_variant: str) -> str:
    ids_text = ", ".join(film.entity_id for film in context)
    if prompt_variant == "direct":
        return (
            f"Among these candidate FILM-IDs: {ids_text}, which FILM-ID has the highest "
            "BOX_OFFICE_MUSD? Return only the winning FILM-ID."
        )
    if prompt_variant == "closed_world":
        return (
            f"Valid candidates: {ids_text}\n"
            "Which valid FILM-ID has the maximum BOX_OFFICE_MUSD among all listed candidates?\n"
            "Read the BOX_OFFICE_MUSD field from the dossiers; do not infer values from titles.\n"
            "Return exactly one valid FILM-ID."
        )
    if prompt_variant == "field_line_max":
        return (
            f"Valid candidates: {ids_text}\n"
            "Compare only the BOX_OFFICE_MUSD integer line for every valid candidate.\n"
            "Which valid FILM-ID has the largest BOX_OFFICE_MUSD?\n"
            "Return exactly one FILM-ID. Do not list values or explain."
        )
    return (
        f"Candidate FILM-IDs: {ids_text}\n"
        "Scan every candidate dossier and compare the BOX_OFFICE_MUSD integers.\n"
        "Which FILM-ID has the highest BOX_OFFICE_MUSD among all candidates?\n"
        "Return only the FILM-ID."
    )


def make_row(
    *,
    trace_index: int,
    context: Sequence[Film],
    chunks_by_id: Dict[str, str],
    chunk_tokens_by_id: Dict[str, Optional[int]],
    first_seen: Dict[str, Dict[str, Any]],
    stage: str,
    scheduled_cell: Optional[str],
    guaranteed_start: Optional[int],
    matrix_threshold: float,
    prompt_variant: str,
    candidate_pair: Optional[Tuple[Film, Film]] = None,
) -> Dict[str, Any]:
    context = list(context)
    context_ids = [film.entity_id for film in context]
    values = {film.entity_id: film.box_office_musd for film in context}
    if candidate_pair is None:
        ordered = sorted(context, key=lambda film: (film.box_office_musd, film.entity_id), reverse=True)
        candidate_left, candidate_right = ordered[0], ordered[1]
    else:
        candidate_left, candidate_right = candidate_pair
    if candidate_left.entity_id not in values or candidate_right.entity_id not in values:
        raise ValueError("candidate_pair must be present in context")
    winner, runner = (
        (candidate_left, candidate_right)
        if candidate_left.box_office_musd > candidate_right.box_office_musd
        else (candidate_right, candidate_left)
    )
    instruction = instruction_for_variant(prompt_variant)
    question = make_question(context, prompt_variant)
    answer = winner.entity_id
    passages = [{"title": film.entity_id, "text": chunks_by_id[film.entity_id]} for film in context]
    ctxs = [chunks_by_id[film.entity_id] for film in context]
    prompt_segments = [instruction] + ctxs + [f"Question: {question}"]
    current_predecessors = {entity_id: context_ids[:idx] for idx, entity_id in enumerate(context_ids)}
    histories = []
    statuses: Dict[str, Optional[str]] = {}
    for film in context:
        cached = first_seen.get(film.entity_id)
        predecessor_ids = [str(item) for item in cached.get("preceding_ids", [])] if cached else []
        predecessor_values = {
            pred_id: first_seen.get(pred_id, {}).get("box_office_musd", values.get(pred_id))
            for pred_id in predecessor_ids
        }
        predecessor_films = [
            Film(
                entity_id=pred_id,
                title="",
                director="",
                release_year=0,
                starlight_awards=0,
                box_office_musd=int(predecessor_values[pred_id]),
                genre="",
                studio="",
                country="",
                runtime_min=0,
            )
            for pred_id in predecessor_ids
            if predecessor_values.get(pred_id) is not None
        ]
        status = status_from_predecessors(film, predecessor_films)
        statuses[film.entity_id] = status
        bigger = sum(1 for value in predecessor_values.values() if isinstance(value, int) and value > film.box_office_musd)
        smaller = sum(1 for value in predecessor_values.values() if isinstance(value, int) and value < film.box_office_musd)
        original_predecessor_relations = [
            {
                "entity_id": pred_id,
                "attribute_value": predecessor_values.get(pred_id),
                "relation_to_current": relation_to_target(predecessor_values.get(pred_id), film.box_office_musd),
            }
            for pred_id in predecessor_ids
        ]
        current_predecessor_relations = [
            {
                "entity_id": pred_id,
                "attribute_value": values.get(pred_id),
                "relation_to_current": relation_to_target(values.get(pred_id), film.box_office_musd),
            }
            for pred_id in current_predecessors.get(film.entity_id, [])
        ]
        histories.append(
            {
                "entity_id": film.entity_id,
                "role": "winner" if film.entity_id == winner.entity_id else ("runner_up" if film.entity_id == runner.entity_id else "nonwinner_candidate"),
                "conditioning_status": status,
                "current_predecessor_ids": list(current_predecessors.get(film.entity_id, [])),
                "current_predecessor_relations": current_predecessor_relations,
                "original_caching_trace_id": cached.get("trace_id") if cached else None,
                "original_caching_query_index_1based": cached.get("query_index_1based") if cached else None,
                "original_caching_predecessor_ids": predecessor_ids,
                "original_caching_predecessor_count": len(predecessor_ids),
                "original_caching_predecessor_attribute_values": predecessor_values,
                "original_caching_predecessor_relations": original_predecessor_relations,
                "original_predecessor_bigger_count": bigger,
                "original_predecessor_smaller_count": smaller,
                "current_attribute_value": film.box_office_musd,
            }
        )
    winner_status = statuses.get(winner.entity_id)
    runner_status = statuses.get(runner.entity_id)
    nonwinner_statuses = [status for entity_id, status in statuses.items() if entity_id != winner.entity_id]
    nonwinner_conditioning_status = None
    if nonwinner_statuses and all(status == nonwinner_statuses[0] for status in nonwinner_statuses):
        nonwinner_conditioning_status = nonwinner_statuses[0]
    computed_cell = None
    if winner_status in {"<", ">"} and nonwinner_conditioning_status in {"<", ">"}:
        computed_cell = f"{winner_status}{nonwinner_conditioning_status}"
    before_seen = set(first_seen)
    all_seen = all(entity_id in before_seen for entity_id in context_ids)
    other_statuses = [status for entity_id, status in statuses.items() if entity_id not in {winner.entity_id, runner.entity_id}]
    other_counts = {"<": other_statuses.count("<"), ">": other_statuses.count(">"), "none": other_statuses.count(None)}
    nonwinner_counts = {
        "<": nonwinner_statuses.count("<"),
        ">": nonwinner_statuses.count(">"),
        "none": nonwinner_statuses.count(None),
    }
    winner_idx = context_ids.index(winner.entity_id)
    runner_idx = context_ids.index(runner.entity_id)
    golden_indices = list(range(len(context)))
    golden_ids = list(context_ids)
    non_golden_indices: List[int] = []
    row = {
        "trace_id": f"onevsall_query_{trace_index:04d}",
        "query_id": f"onevsall_query_{trace_index:04d}",
        "dataset": "one_vs_all_v1",
        "longbench_id": f"one_vs_all_v1_{trace_index:04d}",
        "question": question,
        "answer": answer,
        "answers": [answer, f"Answer={answer}"],
        "answers_all": [answer, f"Answer={answer}"],
        "type": "synthetic_boxoffice_one_vs_all_argmax",
        "n_hops": len(context),
        "num_passages": len(context),
        "passages": passages,
        "ctxs": ctxs,
        "golden_chunk_indices": golden_indices,
        "golden_chunk_titles": [context_ids[idx] for idx in golden_indices],
        "golden_chunk_count": len(golden_indices),
        "non_golden_chunk_indices": non_golden_indices,
        "prompt_segments": prompt_segments,
        "prompt_text": "\n\n".join(prompt_segments),
        "turn_1_poison_prompt": "\n\n".join([instruction] + ctxs),
        "turn_2_eval_prompt": f"Question: {question}",
        "gold_answer": f"Answer={answer}",
        "metadata": {
            "template": "one_vs_all_v1",
            "template_family": "film_attribute_one_vs_all_argmax",
            "comparison_attribute_key": "box_office_musd",
            "comparison_attribute_label": "BOX_OFFICE_MUSD",
            "comparison_attribute_type": "numeric",
            "comparison_direction": "max",
            "question_type": "one_vs_all_argmax",
            "prompt_variant": prompt_variant,
            "instruction_chunk": instruction,
            "context_entity_ids": context_ids,
            "appears_entity_ids": context_ids,
            "candidate_entity_ids": context_ids,
            "winner_entity_id": winner.entity_id,
            "winner_attribute_value": winner.box_office_musd,
            "runner_up_entity_id": runner.entity_id,
            "runner_up_attribute_value": runner.box_office_musd,
            "answer_entity_id": winner.entity_id,
            "answer_chunk_ids": [winner.entity_id],
            "runner_up_chunk_ids": [runner.entity_id],
            "gold_chunk_ids": golden_ids,
            "gold_chunk_roles": {
                film.entity_id: (
                    "winner" if film.entity_id == winner.entity_id else "runner_up" if film.entity_id == runner.entity_id else "nonwinner_candidate"
                )
                for film in context
            },
            "ranked_entity_ids_by_box_office": [
                film.entity_id
                for film in sorted(context, key=lambda item: (item.box_office_musd, item.entity_id), reverse=True)
            ],
            "num_context_chunks": len(context),
            "num_candidates": len(context),
            "pollution_mode": "guaranteed" if guaranteed_start is not None and trace_index + 1 >= guaranteed_start else "warmup",
            "matrix_scheduler_mode": "one_vs_all_v1",
            "matrix_scheduler_stage": stage,
            "scheduled_matrix_cell": scheduled_cell,
            "computed_matrix_cell": computed_cell,
            "winner_conditioning_status": winner_status,
            "runner_up_conditioning_status": runner_status,
            "nonwinner_conditioning_status": nonwinner_conditioning_status,
            "nonwinner_conditioning_status_counts": nonwinner_counts,
            "other_conditioning_status_counts": other_counts,
            "matrix_threshold": matrix_threshold,
            "guaranteed_pollution_start_query": guaranteed_start,
            "all_chunks_seen_before_query": all_seen,
            "comparison_attribute_values": values,
            "chunk_token_counts": {entity_id: chunk_tokens_by_id.get(entity_id) for entity_id in context_ids},
            "pollution_guarantee": {
                "holds": bool(all_seen and scheduled_cell is not None),
                "gold_chunk_ids": golden_ids,
                "answer_chunk_ids": [winner.entity_id],
                "top_two_chunk_ids": [winner.entity_id, runner.entity_id],
                "gold_chunk_histories": histories,
                "top_two_chunk_histories": [item for item in histories if item["entity_id"] in {winner.entity_id, runner.entity_id}],
                "all_chunk_histories": histories,
            },
        },
    }
    return row


def annotate_full_reuse_boundaries(rows: List[Dict[str, Any]], guaranteed_start: int) -> None:
    start_index_0based = guaranteed_start - 1
    for idx, row in enumerate(rows):
        md = row.setdefault("metadata", {})
        is_eval = idx >= start_index_0based
        md["row_index_0based"] = idx
        md["row_index_1based"] = idx + 1
        md["full_reuse_eval_start_index_0based"] = start_index_0based
        md["full_reuse_eval_start_query_1based"] = guaranteed_start
        md["full_reuse_eval_row_index_0based"] = idx - start_index_0based if is_eval else None
        md["full_reuse_eval_row_index_1based"] = idx - start_index_0based + 1 if is_eval else None
        md["is_full_reuse_failure_eval"] = bool(is_eval)


def update_first_seen(row: Dict[str, Any], first_seen: Dict[str, Dict[str, Any]]) -> None:
    context_ids = [str(item) for item in row["metadata"]["context_entity_ids"]]
    values = row["metadata"]["comparison_attribute_values"]
    query_index = int(row["trace_id"].rsplit("_", 1)[-1]) + 1
    for idx, entity_id in enumerate(context_ids):
        if entity_id not in first_seen:
            first_seen[entity_id] = {
                "trace_id": row["trace_id"],
                "query_index_1based": query_index,
                "preceding_ids": context_ids[:idx],
                "box_office_musd": int(values[entity_id]),
            }


def cached_conditioning_maps(first_seen: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, int]]:
    status_by_id: Dict[str, str] = {}
    predecessor_count_by_id: Dict[str, int] = {}
    for entity_id, cached in first_seen.items():
        target_value = int(cached["box_office_musd"])
        predecessor_ids = [str(item) for item in cached.get("preceding_ids", [])]
        predecessor_count_by_id[entity_id] = len(predecessor_ids)
        predecessor_values = [
            first_seen[pred_id]["box_office_musd"]
            for pred_id in predecessor_ids
            if pred_id in first_seen and first_seen[pred_id].get("box_office_musd") is not None
        ]
        if len(predecessor_values) != len(predecessor_ids) or not predecessor_values:
            continue
        if all(int(value) < target_value for value in predecessor_values):
            status_by_id[entity_id] = "<"
        elif all(int(value) > target_value for value in predecessor_values):
            status_by_id[entity_id] = ">"
    return status_by_id, predecessor_count_by_id


def allocate_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    raw = {cell: total * float(ratios.get(cell, 0.0)) for cell in CELL_CODES}
    counts = {cell: int(math.floor(raw[cell])) for cell in CELL_CODES}
    remaining = total - sum(counts.values())
    order = sorted(CELL_CODES, key=lambda cell: (raw[cell] - counts[cell], -CELL_CODES.index(cell)), reverse=True)
    for cell in order[:remaining]:
        counts[cell] += 1
    return counts


def parse_cell_ratios(raw: str) -> Dict[str, float]:
    if not raw.strip():
        return {cell: 1.0 / len(CELL_CODES) for cell in CELL_CODES}
    ratios: Dict[str, float] = {}
    for item in raw.split(","):
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in CELL_CODES:
            raise ValueError(f"Unsupported cell code {key!r}")
        ratios[key] = float(value)
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Cell ratio sum must be positive")
    return {cell: ratios.get(cell, 0.0) / total for cell in CELL_CODES}


def build_eval_context(
    *,
    cell: str,
    cached_films: Sequence[Film],
    status_by_id: Dict[str, str],
    predecessor_count_by_id: Dict[str, int],
    min_gap: int,
    num_chunks: int,
    winner_conditioning_predecessors: int,
    nonwinner_conditioning_predecessors: int,
    rng: random.Random,
) -> Optional[List[Film]]:
    winner_status, nonwinner_status = cell[0], cell[1]
    winner_pool = [
        film
        for film in cached_films
        if status_by_id.get(film.entity_id) == winner_status
        and predecessor_count_by_id.get(film.entity_id, 0) >= winner_conditioning_predecessors
    ]
    nonwinner_pool = [
        film
        for film in cached_films
        if status_by_id.get(film.entity_id) == nonwinner_status
        and predecessor_count_by_id.get(film.entity_id, 0) >= nonwinner_conditioning_predecessors
    ]
    candidates: List[Tuple[Film, List[Film]]] = []
    for winner in winner_pool:
        lower = [
            film
            for film in nonwinner_pool
            if film.entity_id != winner.entity_id
            and winner.box_office_musd - film.box_office_musd >= min_gap
        ]
        if len(lower) >= num_chunks - 1:
            candidates.append((winner, lower))
    if not candidates:
        return None
    winner, lower = rng.choice(candidates)
    lower = sorted(lower, key=lambda film: (abs(film.box_office_musd - winner.box_office_musd), film.entity_id))
    nonwinners = rng.sample(lower[: max(num_chunks - 1, min(len(lower), 32))], k=num_chunks - 1)
    context = [winner] + nonwinners
    rng.shuffle(context)
    return context


def monotonic_warmup_schedule(ordered: Sequence[Film], num_chunks: int) -> List[Tuple[str, List[Film]]]:
    """Return candidate warmup rows ordered from smallest to larger budget.

    A descending row gives each later chunk a `>` prior because all predecessors
    have larger box office.  An ascending row gives each later chunk a `<` prior.
    Rows are targeted rather than simple disjoint rank bands: first create a
    lower `<` support pool, then several high `<` winners, then a wider `>` pool.
    This keeps the warmup small while avoiding one viable winner per sign.
    """
    if num_chunks != 10:
        raise ValueError("one_vs_all_v1 targeted warmup currently requires num_chunks=10")
    if len(ordered) < 50:
        raise RuntimeError(f"Need at least 50 films for one_vs_all_v1 warmup; got {len(ordered)}")

    schedule: List[Tuple[str, List[Film]]] = []
    # Lower `<` support pool.  In each reversed rank slice, all but the first
    # chunk get a `<` status, and most get at least two predecessors.
    schedule.append(("warmup_monotonic_lt_lower_30_39", list(reversed(ordered[30:40]))))
    schedule.append(("warmup_monotonic_lt_lower_40_49", list(reversed(ordered[40:50]))))

    lower_predecessors = list(reversed(ordered[30:39]))
    for rank in range(0, 6):
        schedule.append((f"warmup_target_lt_high_rank{rank:02d}", lower_predecessors + [ordered[rank]]))

    top_predecessors = list(ordered[0:9])
    for rank in range(9, 30):
        schedule.append((f"warmup_target_gt_rank{rank:02d}", top_predecessors + [ordered[rank]]))

    return schedule


def build_eval_rows_for_counts(
    *,
    rows_so_far: Sequence[Dict[str, Any]],
    first_seen: Dict[str, Dict[str, Any]],
    ordered: Sequence[Film],
    cell_counts: Dict[str, int],
    chunks_by_id: Dict[str, str],
    chunk_tokens_by_id: Dict[str, Optional[int]],
    min_boxoffice_gap: int,
    num_chunks: int,
    winner_conditioning_predecessors: int,
    nonwinner_conditioning_predecessors: int,
    min_winners_per_cell: int,
    prompt_variant: str,
    rng: random.Random,
    guaranteed_start: int,
) -> Optional[List[Dict[str, Any]]]:
    status_by_id, predecessor_count_by_id = cached_conditioning_maps(first_seen)
    cached_films = [film for film in ordered if film.entity_id in first_seen and status_by_id.get(film.entity_id) in {"<", ">"}]
    eval_rows: List[Dict[str, Any]] = []
    for cell in CELL_CODES:
        seen_context_keys = set()
        attempts = 0
        while sum(1 for row in eval_rows if row["metadata"].get("scheduled_matrix_cell") == cell) < cell_counts[cell]:
            attempts += 1
            if attempts > max(2000, cell_counts[cell] * 200):
                return None
            context = build_eval_context(
                cell=cell,
                cached_films=cached_films,
                status_by_id=status_by_id,
                predecessor_count_by_id=predecessor_count_by_id,
                min_gap=min_boxoffice_gap,
                num_chunks=num_chunks,
                winner_conditioning_predecessors=winner_conditioning_predecessors,
                nonwinner_conditioning_predecessors=nonwinner_conditioning_predecessors,
                rng=rng,
            )
            if context is None:
                return None
            key = tuple(film.entity_id for film in context)
            if key in seen_context_keys and attempts < 500:
                continue
            seen_context_keys.add(key)
            row = make_row(
                trace_index=len(rows_so_far) + len(eval_rows),
                context=context,
                chunks_by_id=chunks_by_id,
                chunk_tokens_by_id=chunk_tokens_by_id,
                first_seen=first_seen,
                stage="guaranteed_eval",
                scheduled_cell=cell,
                guaranteed_start=guaranteed_start,
                matrix_threshold=0.5,
                prompt_variant=prompt_variant,
            )
            if row["metadata"]["computed_matrix_cell"] != cell:
                return None
            eval_rows.append(row)
        cell_winners = {
            row["metadata"]["winner_entity_id"]
            for row in eval_rows
            if row["metadata"].get("scheduled_matrix_cell") == cell
        }
        if len(cell_winners) < min_winners_per_cell:
            return None
    rng.shuffle(eval_rows)
    return eval_rows


def generate(
    *,
    source_paths: Sequence[Path],
    seed: int,
    num_queries: int,
    num_chunks: int,
    full_hit_rate: float,
    cell_ratios: Dict[str, float],
    eval_per_cell: int,
    max_warmup_rows: int,
    min_boxoffice_gap: int,
    chunk_target_tokens: int,
    tokenizer_path: Optional[Path],
    conditioning_predecessors: int,
    winner_conditioning_predecessors: int,
    nonwinner_conditioning_predecessors: int,
    min_winners_per_cell: int,
    warmup_target_strategy: str,
    prompt_variant: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if num_chunks < 4:
        raise ValueError("num_chunks must be at least 4")
    rng = random.Random(seed)
    films = load_films_from_matrix(source_paths)
    count_tokens = build_token_counter(tokenizer_path)
    chunks_by_id: Dict[str, str] = {}
    chunk_tokens_by_id: Dict[str, Optional[int]] = {}
    for film in films:
        chunk, token_count = long_movie_chunk(film, chunk_target_tokens, count_tokens)
        chunks_by_id[film.entity_id] = chunk
        chunk_tokens_by_id[film.entity_id] = token_count

    ordered = sorted(films, key=lambda film: (film.box_office_musd, film.entity_id), reverse=True)
    cell_counts = {cell: int(eval_per_cell) for cell in CELL_CODES}
    first_seen: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    eval_rows: Optional[List[Dict[str, Any]]] = None
    warmup_schedule = monotonic_warmup_schedule(ordered, num_chunks)
    if max_warmup_rows < len(warmup_schedule):
        raise ValueError(f"max_warmup_rows={max_warmup_rows} is too small; need up to {len(warmup_schedule)} rows")

    for stage, context in warmup_schedule[:max_warmup_rows]:
        row = make_row(
            trace_index=len(rows),
            context=context,
            chunks_by_id=chunks_by_id,
            chunk_tokens_by_id=chunk_tokens_by_id,
            first_seen=first_seen,
            stage=stage,
            scheduled_cell=None,
            guaranteed_start=None,
            matrix_threshold=0.5,
            prompt_variant=prompt_variant,
        )
        rows.append(row)
        update_first_seen(row, first_seen)
        guaranteed_start_candidate = len(rows) + 1
        trial_rng = random.Random(seed * 1009 + len(rows))
        trial_eval_rows = build_eval_rows_for_counts(
            rows_so_far=rows,
            first_seen=first_seen,
            ordered=ordered,
            cell_counts=cell_counts,
            chunks_by_id=chunks_by_id,
            chunk_tokens_by_id=chunk_tokens_by_id,
            min_boxoffice_gap=min_boxoffice_gap,
            num_chunks=num_chunks,
            winner_conditioning_predecessors=winner_conditioning_predecessors,
            nonwinner_conditioning_predecessors=nonwinner_conditioning_predecessors,
            min_winners_per_cell=min_winners_per_cell,
            prompt_variant=prompt_variant,
            rng=trial_rng,
            guaranteed_start=guaranteed_start_candidate,
        )
        if trial_eval_rows is not None:
            eval_rows = trial_eval_rows
            break

    if eval_rows is None:
        status_by_id, predecessor_count_by_id = cached_conditioning_maps(first_seen)
        raise RuntimeError(
            "Could not build requested one_vs_all eval rows within warmup budget; "
            f"warmup_rows={len(rows)} status_counts={status_counts(status_by_id)} "
            f"predecessor_count_by_id={predecessor_count_by_id}"
        )

    guaranteed_start = len(rows) + 1
    for row in rows:
        row["metadata"]["guaranteed_pollution_start_query"] = guaranteed_start

    for idx, row in enumerate(eval_rows, start=len(rows)):
        old_trace = row["trace_id"]
        row["trace_id"] = f"onevsall_query_{idx:04d}"
        row["query_id"] = row["trace_id"]
        row["longbench_id"] = f"one_vs_all_v1_{idx:04d}"
        # Preserve the pre-shuffle construction id for debugging.
        row["metadata"]["pre_shuffle_trace_id"] = old_trace
        row["metadata"]["guaranteed_pollution_start_query"] = guaranteed_start
        row["metadata"]["pollution_mode"] = "guaranteed"

    rows.extend(eval_rows)
    annotate_full_reuse_boundaries(rows, guaranteed_start)
    status_by_id, predecessor_count_by_id = cached_conditioning_maps(first_seen)
    manifest = {
        "seed": seed,
        "num_queries": len(rows),
        "requested_num_queries": num_queries,
        "num_chunks": num_chunks,
        "full_hit_rate": full_hit_rate,
        "eval_per_cell": eval_per_cell,
        "max_warmup_rows": max_warmup_rows,
        "guaranteed_pollution_start_query": guaranteed_start,
        "full_reuse_eval_start_query_1based": guaranteed_start,
        "full_reuse_eval_start_index_0based": guaranteed_start - 1,
        "warmup_rows": guaranteed_start - 1,
        "eval_rows": len(eval_rows),
        "cell_counts": cell_counts,
        "status_counts": status_counts(status_by_id),
        "predecessor_count_by_id": predecessor_count_by_id,
        "chunk_target_tokens": chunk_target_tokens,
        "conditioning_predecessors": conditioning_predecessors,
        "winner_conditioning_predecessors": winner_conditioning_predecessors,
        "nonwinner_conditioning_predecessors": nonwinner_conditioning_predecessors,
        "min_winners_per_cell": min_winners_per_cell,
        "warmup_target_strategy": warmup_target_strategy,
        "prompt_variant": prompt_variant,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
    }
    return rows, manifest


def status_counts(status_by_id: Dict[str, str]) -> Dict[str, int]:
    return {"<": sum(1 for value in status_by_id.values() if value == "<"), ">": sum(1 for value in status_by_id.values() if value == ">")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-queries", type=int, default=80)
    parser.add_argument("--num-chunks", type=int, default=10)
    parser.add_argument("--full-hit-rate", type=float, default=0.5)
    parser.add_argument("--cell-ratios", default="<<=0.25,<>=0.25,><=0.25,>>=0.25")
    parser.add_argument("--eval-per-cell", type=int, default=40)
    parser.add_argument("--max-warmup-rows", type=int, default=40)
    parser.add_argument("--min-boxoffice-gap", type=int, default=80)
    parser.add_argument("--chunk-target-tokens", type=int, default=512)
    parser.add_argument("--conditioning-predecessors", type=int, default=4)
    parser.add_argument("--winner-conditioning-predecessors", type=int, default=9)
    parser.add_argument("--nonwinner-conditioning-predecessors", type=int, default=2)
    parser.add_argument("--min-winners-per-cell", type=int, default=4)
    parser.add_argument("--warmup-target-strategy", choices=["extreme", "stratified"], default="extreme")
    parser.add_argument("--prompt-variant", choices=sorted(PROMPT_VARIANTS), default=DEFAULT_PROMPT_VARIANT)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, default=None)
    args = parser.parse_args()

    rows, manifest = generate(
        source_paths=list(args.source_jsonl) or default_source_paths(),
        seed=args.seed,
        num_queries=args.num_queries,
        num_chunks=args.num_chunks,
        full_hit_rate=args.full_hit_rate,
        cell_ratios=parse_cell_ratios(args.cell_ratios),
        eval_per_cell=args.eval_per_cell,
        max_warmup_rows=args.max_warmup_rows,
        min_boxoffice_gap=args.min_boxoffice_gap,
        chunk_target_tokens=args.chunk_target_tokens,
        tokenizer_path=args.tokenizer_path,
        conditioning_predecessors=args.conditioning_predecessors,
        winner_conditioning_predecessors=args.winner_conditioning_predecessors,
        nonwinner_conditioning_predecessors=args.nonwinner_conditioning_predecessors,
        min_winners_per_cell=args.min_winners_per_cell,
        warmup_target_strategy=args.warmup_target_strategy,
        prompt_variant=args.prompt_variant,
    )
    write_jsonl(args.out, rows)
    manifest_path = args.manifest_out or args.out.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.out}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
