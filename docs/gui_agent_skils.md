## Mobile GUI Agent with Self-Developing Skills

### Vision

- Build a mobile GUI agent that learns new skills from its own successful actions. Example: when the agent plays a track on Spotify via multi-step UI navigation, it automatically distills that trace into a reusable, parameterized "Play Spotify track" skill for instant invocation next time.
- Over time, the agent accumulates a library of reliable skills (tools) that compose into longer workflows, improving speed, reliability, and cost.

### Why now

- Strong LLM planners can operate GUIs with visual + accessibility context but are bottlenecked by repeated long-horizon reasoning and flaky UI steps.
- Recent work demonstrates: (a) self-evolution modules that store tips and shortcuts, (b) skill bootstrapping guided by LLMs, and (c) hierarchical control for open-vocabulary instructions on complex interfaces.

### Related work (non-exhaustive)

- Mobile-Agent-E (self-evolving mobile assistant with Tips/Shortcuts memory) — hierarchical agents that improve via long-term memory.
- BOSS: Bootstrap Your Own Skills — LLM-guided skill discovery and library growth for long-horizon tasks.
- AnySkill — hierarchical open-vocabulary skills via low-level atomic actions + high-level selection with image/text alignment.
- Planning and self-improvement methods: ReAct (reason+act prompting), Reflexion (self-reflection with memory), Toolformer (self-teaching tool use).
- Mobile environments and tooling: Android accessibility services, Appium/uiautomator2, emulator + ADB, environments such as AndroidEnv.

References with links are at the end.

---

### System architecture (Android-first, iOS later)

1) Observation layer
- Accessibility tree: resource-id, content-desc, text, bounds, state.
- Visual context: screenshots; optional OCR for text not exposed via accessibility.
- App state signals: activity name, package, foreground app, network events (optional).

2) Action layer
- Primitive actions: tap(selector), long_tap, swipe, input_text, key(back/home), wait_until(predicate), open_app(package), deep_link(url).
- Execution backends: uiautomator2 or Appium for Android; XCTest/UIAutomation for iOS (later).

3) Planning & execution
- Planner: LLM (ReAct-style) with short-term scratchpad. Chooses actions or calls existing skills.
- Executor: runs chosen actions with timeouts, retries, and guardrails.
- Verifier: checks immediate and end-goal success using assertions and detectors.

4) Skill memory & retrieval
- Vector + symbolic index over skill names, descriptions, app/context tags, and argument schemas.
- Retrieval-augmented planning: planner sees top-k candidate skills before acting.

5) Skill induction (self-development)
- From successful traces: generalize the action sequence into a parameterized, typed skill with preconditions and validators.
- Canonicalize selectors; add robust fallbacks and guards; attach success detectors.
- Versioning, tests, and reliability score before the skill is promoted to default use.

6) Safety & governance
- Scope: per-app allowlists, forbidden actions, rate limits, screenshot redaction.
- Confirmation policy: auto-execute safe skills; require user confirm for risky ones.
- Audit log: actions, screenshots (redacted), selectors, outcomes.

---

### Skill representation (tool schema)

Goals: parameterized, robust to UI changes, testable, explainable.

```yaml
name: play_spotify_track
version: 1.2.0
app: com.spotify.music
description: Play a track by name and optionally artist on Spotify.
arguments:
  - name: track
    type: string
    required: true
  - name: artist
    type: string
    required: false
preconditions:
  - app_installed: com.spotify.music
  - network: online
selectors:
  # Prefer stable identifiers; provide fallbacks.
  search_button:
    primary: { by: id, value: "com.spotify.music:id/search_tab" }
    fallbacks:
      - { by: text, value: "Search" }
      - { by: desc, value: "Search" }
  search_field:
    primary: { by: id, value: "com.spotify.music:id/query" }
  first_result_play:
    primary: { by: desc, value: "Play" }
steps:
  - open_app: com.spotify.music
  - wait_until: { visible: search_button, timeout_ms: 8000 }
  - tap: search_button
  - input_text: { selector: search_field, text: "{{track}} {{artist|default:''}}" }
  - key: enter
  - wait_until: { visible: first_result_play, timeout_ms: 12000 }
  - tap: first_result_play
validators:
  - audio_playing: true
  - ui_contains_text: "{{track}}"
  - app_in_foreground: com.spotify.music
recovery:
  - if: { not: app_in_foreground }
    then: [ { open_app: com.spotify.music } ]
metadata:
  created_from_trace: trace_2025_10_21T12_03Z
  reliability_score: 0.87
  tests_passed: 28
```

Key elements
- Preconditions bound the operating context and allow fast failure.
- Selectors are multi-strategy with fallbacks and prioritization.
- Validators make skills self-checking and idempotent where possible.
- Recovery policies handle common transient failures.

---

### Induction pipeline (turn a successful run into a reusable skill)

1) Trace capture
- Record action sequence, timing, screenshots, accessibility nodes, and outcomes.

2) Segmentation & intent labeling
- Cluster actions into logical substeps (e.g., open app → search → select result → play).
- Infer high-level intent and candidate parameters (track, artist).

3) Selector synthesis and hardening
- Extract stable attributes (resource-id, content-desc) with visual anchors as fallback.
- Generate multiple selector strategies with confidence scores.

4) Parameterization
- Replace literals in actions with template variables; define types and defaults.

5) Preconditions and validators
- Auto-suggest environment checks and end-state assertions (UI + app state + optional signal).

6) Test generation
- Produce synthetic test goals (e.g., multiple tracks/artists), negative cases, and perturbations.
- Run on emulator farms to estimate reliability and latency.

7) Promotion & versioning
- If success rate ≥ threshold and flakiness ≤ threshold, publish as `tool` with semantic version.
- Store provenance and attach a changelog; support canary releases.

8) Continuous refinement (Reflexion-style)
- On failure, capture error context and propose edits to selectors, waits, or validators; re-test.

---

### Planning with skills

- Retrieval-augmented ReAct: for each user instruction, retrieve top-k skills by semantic match and context filters (app, device, locale). The planner chooses between calling a skill with arguments or executing primitives when no suitable skill exists.
- Credit assignment: attribute success/latency to skills vs primitives to drive induction priority and pruning.
- Caching: memoize short action plans for deterministic subgoals.

---

### Success detectors (beyond UI text)

- UI assertions: accessibility text/value/state present; element visible and clickable.
- Visual signals: template match, icon state changes, progress bars.
- App signals: foreground app, activity name, playback state via media session (Android), notifications.
- Heuristic/LLM verification: final screenshot + instruction → "success?" classification with guardrails.

---

### MVP (12–16 days, Android)

Scope
- Single device profile (Pixel emulator, Android 14), English locale.
- Apps: Spotify, YouTube, Settings, Clock (alarms), simple browser.

Deliverables
- Headless runner with: observation (screenshot + accessibility dump), actions (tap/type/swipe/back), and logging.
- LLM planner (ReAct-style) with a small built-in toolset and guardrails.
- Trace recorder and skill compiler producing YAML tools as above.
- Skill registry (JSON/SQLite) + retrieval index (FAISS or on-disk embeddings).
- Verifier with basic UI assertions + screenshot-based LLM fallback.
- 15–20 scripted evaluation tasks with reproducible seeds.

Milestones
- D1–D3: Device harness (adb + uiautomator2), observation/action APIs.
- D4–D6: Planner + executor + verifier loop; logging and rollbacks.
- D7–D9: Trace capture → skill compiler v1; parameterization and selectors.
- D10–D12: Test runner, reliability scoring, promotion pipeline.
- D13–D16: App coverage expansion, usability polish, safety prompts.

Metrics
- Task success rate, median steps, median wall time, human confirmations needed, skill reuse ratio, flake rate (rerun inconsistency), and time-to-skill (first success → promoted tool).

---

### Risks and mitigations

- UI drift and A/B experiments: multi-strategy selectors + periodic re-tests; anchor to semantics, not pixels.
- Internationalization: tag skills with locale; parameterize text; prefer IDs over visible text.
- Auth walls and deep links: preconditions and prerequisite skills (login); safe storage of secrets; explicit scope.
- Safety/privacy: redaction, on-device inference where possible; user consent gates for risky actions.
- Cost/latency: prefer skills over long chains; on-device small models for verification; batch emulator testing.

---

### Brainstorming: extensions and product ideas

- Skill marketplace and sharing: federated distillation of common skills across users/devices; privacy-preserving analytics.
- Programmatic APIs: export skills as OpenAPI-like tool specs so non-GUI agents can call them.
- Auto-healing skills: online shadow runs of alternative selectors; pick the most reliable live.
- Hierarchical meta-skills: compose verified skills into routines ("commute playlist": open maps, check traffic, start playlist).
- Safety modes: "dry run" visualized playback; time-bounded execution; per-app sandboxes.

---

### References

- Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks — arXiv: `https://arxiv.org/abs/2501.11733`
- BOSS: Bootstrap Your Own Skills — arXiv: `https://arxiv.org/abs/2310.10021`
- AnySkill: Learning Open-Vocabulary Physical Skill for Interactive Agents — arXiv: `https://arxiv.org/abs/2403.12835`
- Also related: ReAct (reason+act prompting), Reflexion (self-improvement with memory), Toolformer (self-teaching tool use), AndroidEnv (mobile RL environment), Appium/uiautomator2 (mobile UI automation).


