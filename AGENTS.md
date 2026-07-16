# AGENTS.md

You operate in a strict read-write feedback loop using the root `scratchpad.md` file.

1. INITIALIZATION: Your very first action in every workspace session must be to read `scratchpad.md` to rehydrate state and grab your target goals.
2. EXECUTION DISCIPLINE: You can *not* change this `AGENTS.md` file without permission request. Agentic AI assistants contributing to this repository must comply with the project policies in `AGENTS.md`.
3. STATE PERSISTENCE: Before completing a task or prompting for user review, overwrite `scratchpad.md` with updated status checklist values, state metrics, or new blockers found.
4. FRESHNESS OVER APPEND: Overwrite or modify the checklist items. Do not append infinitely growing history logs to `scratchpad.md`. Keep it concise.


## Project Context
<!-- Provide a 1-2 sentence high-level description of what this project actually is. -->
- Problem Name: Classification of histopathology tissue images of the  LC25000 dataset.
- Objective: Classify histopathology tissue images of the  LC25000 dataset using implemented and compared Deep Learning models
- Classes: colon adenocarcinoma, benign colon tissue, lung adenocarcinoma, lung squamous cell carcinoma, and benign lung tissue.
- Scope: This project is intended for research and educational purposes only.
- Important: This is **not** a clinical product and must not be used for medical diagnosis or clinical decision-making. 

## Core Stack & Architecture
<!-- Detail critical architecture constraints the agent cannot infer just by reading files. -->
- Keep backend modular. 
- Prototype: nicegui
- Separate inference from UI. 
- Models management:  each model independently
- Minimalistic interface.
<!-- Never hardcode paths. -->

## Executable Commands
<!-- Give exact terminal strings for tasks. Agents copy-paste these verbatim. -->
- Build: ``
- Format & Lint Check: ``
- Test Runner: ``
- Integration Tests: ``

## Code Style & Conventions
<!-- Do not use vague terms like "write clean code". Use specific engineering constraints. Examples
- Language: Strict TypeScript. Never use `any`.
- Modules: Prefer ES modules and explicitly named exports over default exports.
- Components: Strictly functional components with React Hooks.
- Styling: Use atomic Tailwind utility classes; do not write custom CSS or style tags. -->
- Prefer readability over clever code.
<!--Document every public function. -->

## Git & PR Workflow
<!-- Instruct the AI how to format its contributions. -->
- Commit Pattern: Follow Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`).
- Branching: Follow the Git Flow model using five dedicated branch types to isolate work:
  - `main`: Production-ready code only; updated via release/hotfix merges and tagged with version numbers.
  - `develop`: The primary integration branch; all completed features merge here for ongoing development.
  - `feature/*`: Created from `develop` to isolate new tasks; merged back into `develop` via PR when complete.
  - `release/*`: Created from `develop` to polish upcoming versions; merges into both `main` and `develop`.
  - `hotfix/*`: Created directly from `main` to patch urgent production bugs; merges into both `main` and `develop`.


## Boundaries & Constraints
<!-- Define absolute "no-go" zones to stop the agent from dangerously modifying code. -->
<!--- NEVER auto-upgrade package versions in `package.json` without explicit request. -->
- NEVER modify files inside the `docs` directory.
- NEVER auto-upgrade package versions without explicit request.
- Stop and ask the human user if a refactor impacts more than 3 distinct files.
