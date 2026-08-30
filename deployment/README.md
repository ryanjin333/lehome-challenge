# B1K deployment smoke campaign

`b1k-deploy smoke-campaign` is dry-run-first. Without `--execute`, it makes
zero Vast, Docker, Hugging Face, or SSH calls and writes no ledger or receipt.

An executed campaign requires exact digest-qualified image releases, numeric
template IDs, payload hashes, a private SSH identity, a campaign-local
`known_hosts`, the local Vast API-key file boundary, and the account Hugging
Face token file. It runs training before rollout, preserves the USD 5.00 ledger
cap, and destroys only the recorded exact instance ID in `finally`.

The current Vast template UI configuration must independently prove private
Docker pull authentication before execution. Set
`B1K_VAST_PRIVATE_PULL_READY=verified` only after that provider-side proof has
been recorded; the CLI cannot inspect or infer hidden UI credentials. A
rollout no-op infrastructure quarantine is not a policy success and must not
be reported as one.

## Local and CI verification

Run deployment checks with wheel-installed local packages so the frozen
environment does not depend on editable `.pth` processing:

```bash
uv lock --check --project deployment
uv run --project deployment --frozen --no-editable python -c "import b1k_deploy, b1k_rollout"
uv run --project deployment --frozen --no-editable pytest deployment/tests -q
```

`publish-campaign` must receive `--source-root` for the checked-out source
workspace. The command reads both canonical template files from that root and
uses the same root for configured Docker builds; it never infers templates
from the installed package location.
