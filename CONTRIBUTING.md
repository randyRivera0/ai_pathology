# Contributing to AI Pathology

Thank you for contributing. This repository is a research and educational
project, not a clinical product. Contributions must not present model output as
medical advice or support clinical diagnosis or decision-making.

## Development setup

From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the frontend with:

```powershell
python frontend\main.py
```

## Branching workflow

The repository follows Git Flow:

- `main` contains production-ready releases.
- `develop` integrates completed work for the next release.
- `feature/*` branches start from `develop` and merge back through pull requests.
- `release/*` branches prepare a version for `main` and are merged back into `develop`.
- `hotfix/*` branches start from `main` and are merged into both `main` and `develop`.

Start ordinary work from the latest `develop`:

```powershell
git switch develop
git pull --ff-only
git switch -c feature/<short-description>
```

Do not commit ordinary feature work directly to `main` or `develop`.

## Commits

Use Conventional Commits with a short imperative description:

```text
feat: add image upload preview
fix: reject unsupported image formats
docs: document frontend setup
chore: configure development tooling
```

Keep commits focused. Do not combine unrelated formatting, refactoring, and
behavior changes in one commit.

## Pull requests

Push the feature branch and open a pull request into `develop`:

```powershell
git push -u origin feature/<short-description>
```

Each pull request should explain:

- what changed and why;
- which user workflow or research objective it supports;
- how the change was validated;
- known limitations or follow-up work;
- screenshots for visible interface changes.

Keep pull requests small enough to review. Resolve review feedback on the
feature branch rather than committing directly to the target branch.

### Current enforcement status

This private repository does not currently have access to enforced GitHub
rulesets. Until the repository moves to GitHub Pro or an organization on Team,
the following requirements are mandatory project policy but are not all
technically enforced by GitHub:

- open a pull request for every change to `develop` or `main`;
- request the owners listed in `.github/CODEOWNERS` for review;
- do not push directly to `develop` or `main`;
- resolve review conversations before merging;
- use rebase merge to maintain linear history;
- delete merged feature branches;
- never force-push or delete `develop` or `main`.

When enforced rulesets become available, configure them to match this policy.

### Deferred issue forms

`CODEOWNERS` and the pull-request template provide enough governance for the
current project size. If collaborator or issue volume grows, add structured
YAML forms under `.github/ISSUE_TEMPLATE/` for bug reports, feature requests,
and research tasks, together with a `config.yml` file for the issue chooser.

Issue forms become valuable when free-form reports make triage inconsistent.
They should require the information needed for actionable work, such as
reproduction steps and environment details for bugs, acceptance criteria for
features, and hypotheses, datasets, metrics, and expected outputs for research
tasks.

## Releases

Create a release branch from an updated `develop` when the integrated work is
feature-complete and ready for stabilization:

```powershell
git switch develop
git pull --ff-only
git switch -c release/v1.0.0
git push -u origin release/v1.0.0
```

Release branches may contain only release preparation: bug fixes,
documentation, version metadata, and final validation changes. Do not add new
features after creating the branch.

Open a pull request from `release/v1.0.0` into `main`. After it passes review
and CI, merge it and tag the resulting commit on `main`:

```powershell
git switch main
git pull --ff-only
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

Also merge the release branch back into `develop` through a separate pull
request so release fixes are preserved in future work. Delete the release
branch after both merges are complete.

## Hotfixes

Use a hotfix only for an urgent correction to the version currently on `main`.
Create it from an updated `main`:

```powershell
git switch main
git pull --ff-only
git switch -c hotfix/<short-description>
git push -u origin hotfix/<short-description>
```

Keep the change narrowly focused. Open a pull request into `main`, validate and
merge it, then create a patch tag such as `v1.0.1`. Open a second pull request
from the same hotfix branch into `develop` so the fix is not lost from the next
release. Delete the hotfix branch after both merges are complete.

Do not merge release or hotfix branches directly from a local checkout. Use
reviewed pull requests and require the relevant CI checks to pass.

## CI/CD convention

CI configuration belongs in `.github/workflows/` and should be treated as
version-controlled application infrastructure. At minimum, pull requests into
`develop` and `main` should run formatting, linting, tests, and a frontend
startup check once those commands exist.

Use this separation:

- CI validates every pull request and must not publish or deploy artifacts.
- CD runs only from an approved release tag such as `v1.0.0`.
- Deployment environments and secrets are configured in GitHub Environments,
  never committed to the repository.
- Production deployment requires environment approval and must use the exact
  artifact already validated by CI.
- Failed checks block merging; failed deployments do not trigger an automatic
  code rollback unless a tested rollback mechanism exists.

The contributor guide defines these rules. Add a dedicated deployment runbook
only when the project has a real hosting target, environments, secrets,
rollback process, and an assigned maintainer. That runbook should document
deployment and recovery operations, while workflow YAML remains the executable
source of truth.

## Quality and architecture

- Prefer readable implementation over clever abstractions.
- Keep inference logic separate from the NiceGUI interface.
- Keep the backend modular and manage each model independently.
- Define paths through configuration; never hardcode machine-specific paths.
- Keep the five LC25000 class labels and their model-output order in one shared
  source of truth.
- Do not modify files in `docs/` through automated agent contributions.
- Do not upgrade dependency versions without explicit approval.

Run all available checks before requesting review. Until dedicated automated
commands are added, at minimum start the application and manually verify the
workflow affected by the change.

## Data and model artifacts

Do not commit datasets, patient information, credentials, local virtual
environments, large model weights, or generated caches. Confirm licensing and
privacy requirements before adding any external data or model artifact.
