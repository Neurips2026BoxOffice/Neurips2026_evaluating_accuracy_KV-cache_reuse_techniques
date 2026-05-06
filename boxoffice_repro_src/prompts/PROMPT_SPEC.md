# Prompt specification

This bundle does not use a separate chat-style system prompt field. The prompt
begins with a single instruction segment, followed by ten dossier chunks,
followed by the question segment.

## Instruction segment

```text
Use only the synthetic movie dossiers below. Ignore outside/world knowledge. For ranking, compare only the BOX_OFFICE_MUSD integer fields across all named candidates. Return exactly one FILM-ID and no other text.
```

A plain-text copy is also stored in `INSTRUCTION_CHUNK.txt`.

## Question template

```text
Valid candidates: FILM-XXXX, FILM-XXXX, FILM-XXXX, FILM-XXXX, FILM-XXXX, FILM-XXXX, FILM-XXXX, FILM-XXXX, FILM-XXXX, FILM-XXXX
Which valid FILM-ID has the maximum BOX_OFFICE_MUSD among all listed candidates?
Read the BOX_OFFICE_MUSD field from the dossiers; do not infer values from titles.
Return exactly one valid FILM-ID.
```

A plain-text copy is also stored in `QUESTION_TEMPLATE.txt`.

## Prompt assembly order

Each eval prompt has 12 segments in this order:

1. instruction segment
2. dossier chunk 1
3. dossier chunk 2
4. dossier chunk 3
5. dossier chunk 4
6. dossier chunk 5
7. dossier chunk 6
8. dossier chunk 7
9. dossier chunk 8
10. dossier chunk 9
11. dossier chunk 10
12. question segment

## Where this information appears in the JSONL rows

For each produced row in `reference_inputs/*jsonl`:

- `question`: the exact question text for that row
- `metadata.instruction_chunk`: the exact instruction segment
- `prompt_segments`: the 12 prompt segments in order
- `prompt_text`: the fully concatenated prompt

## Example note

An example row in the shared filtered release has question text of the form:

```text
Valid candidates: FILM-2011, FILM-1015, FILM-1029, FILM-1024, FILM-1009, FILM-2043, FILM-2025, FILM-1018, FILM-1028, FILM-1012
Which valid FILM-ID has the maximum BOX_OFFICE_MUSD among all listed candidates?
Read the BOX_OFFICE_MUSD field from the dossiers; do not infer values from titles.
Return exactly one valid FILM-ID.
```
