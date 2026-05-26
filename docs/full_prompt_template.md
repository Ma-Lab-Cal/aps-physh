# Full PhySH prompt template

Prompt template from the paper appendix. In the production run, `{exemplars_json}` was replaced by a batch of 1,000 curated PhySH-labeled examples, `{valid_disciplines_json}` and `{valid_concepts_json}` by the complete canonical PhySH label lists, and `{items_json}` by a batch of 50 papers to label.

```text
You perform rigorous, reproducible labeling for physics subject headings.
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

Examples (gold):
{exemplars_json}

Use only these valid disciplines:
{valid_disciplines_json}

Use only these valid concepts:
{valid_concepts_json}

User:
{items_json}
```
