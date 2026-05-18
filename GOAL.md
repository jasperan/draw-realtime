<goal>
Improve draw-realtime through an autonomous quality loop that makes the project easier to validate, safer to operate, and simpler to maintain without changing model output behavior or frontend product scope.
</goal>

<context>
Start from the repository root. Read these files first:
- README.md
- requirements.txt
- requirements-dev.txt
- scripts/verify.sh
- app/main.py
- app/video_source.py
- app/video_processor.py
- app/pipeline.py
- app/multistyle.py
- app/monarchrt_pipeline.py
- app/quantization/utils.py
- tests/conftest.py
- tests/test_api.py
- tests/test_video_source.py
- frontend/package.json

Useful discovery commands:
- git status --short
- rg -n "TODO|FIXME|HACK|except Exception|pass$|torch\.load|urlopen|0\.0\.0\.0" app tests frontend/src README.md requirements.txt
- python3 -m compileall app tests
- ./scripts/verify.sh
- npm audit --omit=dev --audit-level=high --prefix frontend
</context>

<constraints>
- Preserve unrelated user changes and generated artifacts.
- Keep each improvement narrow, reviewable, and reversible.
- Do not change default model behavior, prompts, inference quality, quantization math, or GPU/runtime assumptions unless the change is explicitly isolated and verified.
- Do not force breaking dependency upgrades without first documenting impact and proving the app still builds.
- Do not hide security findings by suppressing tools or weakening checks.
- Prefer tests, docs, and local helper scripts that improve repeatability before broad refactors.
- For security-sensitive changes, add or update focused regression tests before editing production code when feasible.
- Treat missing CUDA, TensorRT, torch, or heavyweight ML dependencies as environment blockers; document them instead of faking success.
</constraints>

<done_when>
Complete one autonomous improvement cycle when all of the following are true:
- At least one concrete project improvement is implemented in tracked files.
- The improvement is tied to one of these lanes: reproducible validation, test coverage, security hardening, dependency posture, documentation accuracy, or dead-code/duplication cleanup.
- Focused tests or checks for the touched area pass, or an environment blocker is documented with the exact failing command.
- ./scripts/verify.sh has been run and its pass/fail/skip summary is reported.
- npm run build passes when frontend files or lockfiles are touched.
- npm audit --omit=dev --audit-level=high --prefix frontend reports no production high/critical vulnerabilities, or any finding is documented with remediation.
- python3 -m compileall app tests passes.
- git diff --check passes.
- A code review pass is performed on the final diff, with accepted findings fixed or explicitly rejected with rationale.
- A security review pass is performed for the touched surface, including secrets scan and relevant dependency/static checks.
- The final response lists changed files, verification evidence, remaining risks, and the next highest-value improvement.
</done_when>

<workflow>
1. Inspect git status and preserve any unrelated changes.
2. Review this GOAL.md and the context files listed above.
3. Build an improvement shortlist from local evidence: failing checks, security scan output, duplicated code, weak tests, stale docs, or dependency advisories.
4. Pick the highest-value narrow item that can be completed and verified in this cycle.
5. Write or update focused tests/checks before production edits when practical.
6. Implement the smallest production/doc/tooling change that satisfies the selected item.
7. Run focused verification, then ./scripts/verify.sh.
8. Run code review and security review on the final diff.
9. Fix actionable review findings and rerun affected checks.
10. Stop when the done_when contract is satisfied for this cycle; do not widen scope into a second unrelated improvement unless the first cycle is already complete and the user explicitly asked for more cycles.
</workflow>

<verification_loop>
Run these checks after meaningful edits:
- python3 -m compileall app tests
- git diff --check
- ./scripts/verify.sh

Run these checks when relevant:
- python3 -m pytest tests/test_api.py tests/test_video_source.py
- npm run build --prefix frontend
- npm audit --omit=dev --audit-level=high --prefix frontend
- python3 -m pip_audit -r requirements.txt or pip-audit equivalent when available
- bandit -q -r app when available

If a command cannot run because dependencies are missing, record the exact command, error, and environment assumption needed to unblock it.
</verification_loop>

<execution_rules>
- Check git status before edits.
- Preserve unrelated user changes.
- Prefer rg over grep when available.
- Use the runtime's patch/edit tool for manual edits when available.
- Read context files before implementation.
- Batch independent file reads in parallel when the runtime supports it.
- Run focused tests before broad tests.
- Do not paper over failures.
- Do not widen scope.
- Keep the final answer concise.
</execution_rules>

<output_contract>
At completion, provide:
- One-line outcome.
- Changed files.
- Verification commands and results.
- Code review and security review status.
- Remaining risks or blocked checks.
- Recommended next /goal prompt or next cycle target.
</output_contract>
