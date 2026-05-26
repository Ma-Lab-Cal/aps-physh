from __future__ import annotations

import json
import os
from typing import Any

from google import genai
from google.genai import types

from physh_tools.physh import concept_labels, discipline_labels

DEFAULT_MODEL = "gemini-2.5-flash-lite-preview-09-2025"

CLASSIFICATION_INSTRUCTIONS = """You perform rigorous, reproducible labeling for physics subject headings.
Use canonical PhySH names; normalize common synonyms to the canonical form.
Identify the core problem and method from title and abstract before labeling.
Prefer clarity over verbosity; minimize non-essential wording.
Resolve ambiguity by preferring labels supported by explicit keywords.
When a method is emphasized, include an appropriate Techniques concept.
Resolve conflicts by deferring to precise technical usage over colloquial usage.

Prefer a single discipline when one field clearly dominates the core contribution.
Assign a second discipline when the paper makes a distinct, independently sufficient contribution to that field; do not under-assign if two fields are central.
If the paper develops or validates a method that is itself a central contribution, and not merely applied, include the method's discipline as a second discipline.
Rarely, assigning three disciplines is permitted if the paper fundamentally bridges three distinct fields with equal weight.
If three fields are equally central and independently justified, include all three; do not collapse to two.
The threshold for assigning a second or third discipline is high: the paper must advance the state of the art in each field.

When a paper spans multiple disciplines, verify that each assigned discipline is essential to the paper's core contribution.
If one is merely a context or tool, exclude the discipline, but retain specific concepts that accurately describe that method or tool.
Prioritize the single best-fitting discipline when evidence is insufficient for multiple fields.
If the paper applies established methods from one field to a problem in another, assign only the discipline of the problem being solved.
However, if this filtering results in zero disciplines, assign the single most relevant discipline representing the paper's general context.

Critical instruction: list concepts in order from broadest to most specific.
Start the labeling process with the major field, then the subfield, then the specific topic.
For example, use an ordering such as "Condensed matter physics," then "Semiconductors," then "Quantum dots."
Explicitly including broad terms first ensures that they are not omitted.
Make sure the broad terms are present in the final JSON list.

Critical instruction: balance specificity and frequency.
First, assign a Property concept only if the paper investigates the nature or mechanism of that property.
If the property is only a known feature of the system, tag the system rather than the property.
Second, include broad Physical Systems labels for every specific system mentioned.
For example, if "Graphene" is assigned, then "2-dimensional systems" must also be assigned.
Third, for Techniques concepts, prefer standard, established method names, such as "Monte Carlo methods."
Fourth, for Research Areas, ensure that the broad research area is present when supported by the title and abstract, such as "Electronic structure."

Critical instruction: distinguish concepts from disciplines.
Do not simply repeat the discipline name as a concept.
In particular, "Condensed Matter, Materials & Applied Physics," "Particles & Fields," "Atomic, Molecular & Optical," and "Gravitation, Cosmology & Astrophysics" are disciplines, not valid concepts.
Do not assign them as concepts.
Concepts must describe specific topics, systems, properties, techniques, or research areas within the field.

Critical instruction: apply the discipline assignment rules conservatively.
"Quantum Information" and "Statistical Physics" are often used as tools in Condensed Matter.
If the paper makes a substantive contribution to Quantum Information or Statistical Physics methods or theory, rather than merely applying them, assign the corresponding discipline.
If the paper applies standard Quantum Information or Statistical Physics tools to a Condensed Matter system, the discipline is only "Condensed Matter, Materials & Applied Physics."

Represent both main phenomena and relevant theoretical or experimental frameworks when clearly described.
Normalize morphological variants to the canonical PhySH label.
Do not return empty discipline or concept lists; for each paper, return at least one discipline and at least one concept.
Maintain consistency with reference-year distributions when evidence allows, without forcing labels.
Resolve time-related ambiguity by preferring labels consistent with the likely publication era.
Avoid patterns that would systematically inflate rare concepts without strong evidence.
Do not invent new labels or paraphrase existing ones.
Adhere to the allowed labels without deviation.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "disciplines": {"type": "array", "items": {"type": "string"}},
                    "concepts": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "disciplines", "concepts"],
                "propertyOrdering": ["id", "disciplines", "concepts"],
            },
        }
    },
    "required": ["results"],
    "propertyOrdering": ["results"],
}


def build_prompt(items: list[dict[str, Any]], exemplars: list[dict[str, Any]] | None = None) -> str:
    payload = [
        {"id": i, "text_input": item["text"]}
        for i, item in enumerate(items)
    ]
    prompt_parts = [CLASSIFICATION_INSTRUCTIONS]
    if exemplars:
        prompt_parts.extend(
            [
                "Examples (gold):",
                json.dumps(exemplars, ensure_ascii=False),
            ]
        )
    prompt_parts.extend(
        [
            "Use only these valid disciplines:",
            json.dumps(discipline_labels(), ensure_ascii=False),
            "Use only these valid concepts:",
            json.dumps(concept_labels(), ensure_ascii=False),
            "User:",
            json.dumps(payload, ensure_ascii=False),
        ]
    )
    return "\n".join(prompt_parts)


def classify_items(
    items: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 1,
    exemplars: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not items:
        return []
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Copy .env.example to .env or export it.")

    prompt = build_prompt(items, exemplars=exemplars)
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        response_mime_type="application/json",
        response_json_schema=SCHEMA,
    )
    response = client.models.generate_content(model=model, contents=prompt, config=config)
    payload = json.loads(response.text or "{}")
    by_id = {int(row.get("id", -1)): row for row in payload.get("results", []) if isinstance(row, dict)}

    valid_disciplines = set(discipline_labels())
    valid_concepts = set(concept_labels())
    output: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        row = by_id.get(i, {})
        disciplines = [d for d in row.get("disciplines", []) if d in valid_disciplines]
        concepts = [c for c in row.get("concepts", []) if c in valid_concepts]
        output.append(
            {
                **{k: v for k, v in item.items() if k != "text"},
                "text": item["text"],
                "predicted_disciplines": disciplines,
                "predicted_concepts": concepts,
            }
        )
    return output
