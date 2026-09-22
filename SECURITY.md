# Security policy

## Reporting

Report vulnerabilities through GitHub private vulnerability reporting on [planeon-ai/jev-xai](https://github.com/planeon-ai/jev-xai). Do not open a public issue for a security report.

## What this library does not do

jev-xai does not phone home. It makes no outbound network call except the model calls you configure. Logs never include raw decision inputs at INFO, because those payloads are often prompts and other personal data.

Evidence packs can contain those inputs when `redaction_mode` is `full`. Use `redacted` or `hash_only` when the pack will leave the machine that produced it.
