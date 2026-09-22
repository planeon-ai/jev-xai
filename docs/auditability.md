# Auditability

`jev-xai audit decision.json --model module:factory --out audit/` writes:

```text
audit/
├── decision.json
├── explanation.json
├── counterfactuals.json
├── ablation.json
├── stability.json
├── replay.json
├── manifest.json
├── report.md
└── report.html
```

`manifest.json` lists member hashes, a Merkle root, the library version, `schema_version`, `config_hash`, and the limitations text. `verify_pack` recomputes the root. Tampering with a member fails verification.

Objects in the store are addressed by blake2b of canonical JSON. Writes use a temporary file and `replace`.

Redaction:

- `full` stores the input
- `redacted` replaces named fields with `[REDACTED]` (the input hash is still of the original)
- `hash_only` stores no payload. Perturbation explainers refuse the record.

The HTML report is one file, inline CSS, no JavaScript. Bars below the noise floor are grey. Interactive dashboards are out of scope.
