# Record format

`schema_version` is `1.0.0`. Readers accept any `1.x` record and reject other major versions with `SchemaVersionError`.

Canonical JSON: sorted keys, no insignificant whitespace, Unicode NFC strings, floats rendered with `format(value, ".17g")`. `input_hash` is blake2b of that form. The hash does not depend on dict insertion order.

JSON Schema documents generated from the pydantic models:

- `schemas/decision_record.v1.json`
- `schemas/evidence_pack.v1.json`

The code schema is the source of truth. Regenerate the files when the models change; a test compares them.

`container_digest` and `hardware` are best-effort and never gate replay.

Trace fields (`trace_id`, `span_id`, `agent_id`, `step_index`, `parent_decision_id`) are part of the V1 record so a decision inside an agent loop can be identified later. Framework adapters that emit those fields can wait.
