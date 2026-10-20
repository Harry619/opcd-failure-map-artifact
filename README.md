# opcd-failure-map-artifact
Code and data for the paper.


# Artifact: When Does Iterated Context Distillation Break? — A Failure Map, Entropy Signatures, and Guardrails

This repository contains the complete experimental system behind the paper: drivers for every experiment family (failure-map probes, guard arms, dose controls, the math-8B refresh chain), the modified OPCD/verl training framework, per-experiment status logs, the analysis scripts that produce the paper's tables, and the pre-registered design documents.

## Repository layout

```
opcd-failure-map-artifact/
├── README.md                  <- this file
├── code/
│   ├── drivers/               <- all experiment drivers (bash), by family
│   │   ├── e1_probes/         (14) E1 failure-probe family: FL/SK x static/refresh
│   │   │                            probes, pool autopsies, context-recovery tests
│   │   ├── p_probes/          (16) P1–P11 targeted probes (greedy agreement,
│   │   │                            context re-injection, per-env scoring patches)
│   │   ├── e2_guards/         (10) guard arms: entropy-gate (A), rollback (B),
│   │   │                            combined (C, incl. second seed G8), blind (E),
│   │   │                            adaptive + G9 bandwidth sensitivity
│   │   ├── b_series/          ( 7) dose controls: B1 schedule-vs-dose, B2 lr/2,
│   │   │                            B3 re-evaluations
│   │   ├── b6_math8b/         ( 7) the math-8B refresh chain: 6-version driver
│   │   │                            history (v6 final), checkpoint-resume driver,
│   │   │                            early-signal collectors, final eval
│   │   ├── evaluation/        (11) shared evaluation drivers (k-seed re-vals,
│   │   │                            cross-machine reproductions, phase evals)
│   │   └── infrastructure/    ( 6) while-guard self-healing wrapper, memory
│   │                                scheduler, GPU guards, auto-launchers
│   ├── framework/             <- the OPCD framework (verl-based) with all
│   │                              in-tree patches used by the drivers above
│   └── analysis/              ( 3) result-table generators: three-rule
│                                  (Avg@k/Maj@k/Pass@k) scoring, bootstrap CIs,
│                                  chain-health table
├── logs/                      (14) verbatim per-family status logs (driver-level
│                                  RESULT/CHAIN/ROLLBACK lines; large ones trimmed
│                                  to the final segments)
├── results/
│   ├── all_results_summary.md  <- every paper table as a flat markdown summary
│   ├── theta_e1_final_report.md <- the full E1 analysis report (data source)
│   └── b4_results.md           <- three-rule per-env scoring full table
└── docs/
    ├── paper_draft.pdf          <- the submitted PDF (anonymized)
    ├── theta_e1_final_report.md <- E1 failure map: design + results + claims
    ├── theta_e1_frozenlake_iteration.md <- E1 protocol details
    ├── theta_e2_design.md        <- E2 guard design (pre-registered criteria)
    ├── theta_inflight_experiments_design.md <- E2a/B/C/E arm details
    ├── theta_prereg_tier12_20261006.md <- B1/B2 dose pre-registration
    ├── theta_gap_analysis_vs_two_iclr_papers.md <- design-influence audit
    ├── theta_idea_origin.md      <- the "why" narrative
    └── reproduction_results.md   <- OPCD reproduction fidelity table
```

## Experiment map

| Paper claim | Family | Key drivers | Status log |
|---|---|---|---|
| Failure map (2x2x2: domain x scale x regime) | e1_probes | `theta_e1_fl.sh`, `theta_e1b_sk.sh`, … | (per-probe lines in docs/) |
| Weight-level irreversibility (no contextual repair) | p_probes | `theta_p1_probe.py`, `theta_p11_driver.sh` | — |
| Three failure modes + entropy signatures | e2_guards | `theta_e2a_guard.sh` (gate calibration) | `theta_e2aG9{lo,hi}_status.log` |
| Guard arms A/B/C/E + second seed | e2_guards | `theta_e2aC_combo_h20.sh`, `theta_e2aE_blind_h20.sh`, `theta_e2aG8_h20.sh` | `theta_e2aC/E/G8_status.log` |
| Dose, not schedule | b_series | `theta_b1_dose.sh`, `theta_b2_lr.sh` | `theta_b1/b2*_status.log` |
| Math-8B refresh chain (third domain) | b6_math8b | `theta_b6_v6_h20.sh` → `theta_b6_resume12.sh` | `theta_b6v6_status.log`, `theta_b6_early_status.log` |
| Cross-machine evaluation equivalence | evaluation | `theta_r5rt_*`, `theta_m1*_h20.sh` | `theta_r5rt_eval_status.log` |
| Reproduction fidelity | (pipeline) | see `docs/reproduction_results.md` | — |
| Band sensitivity (G9) | e2_guards | `theta_e2aG9{lo,hi}_h20.sh` | embedded in results summary |

Unattended operation: every long chain is wrapped by `code/drivers/infrastructure/theta_while_guard.sh` (relaunch-on-death with disk-marker resume), and the GPU-memory scheduler sequences multi-phase jobs; these are part of the artifact because the failure map required~200 GPU-days of chains that had to self-heal.

## Reproduction quickstart

Hardware used: one node, 2x 80GB-class GPUs per dual-card job (4 GPUs total for the full queue); ~2TB RAM recommended with CPU offloading enabled.
Software: the conda environment is standard verl/vllm stack (see `code/framework/` for entry points; key deps: torch 2.x, vllm 0.8.x, ray, transformers).

1. Train one static chain (FrozenLake 1.7B, e.g. E1): `bash code/drivers/e1_probes/theta_e1_fl.sh` — the driver waits for free GPUs, runs consolidate→eval per round, and appends RESULT lines to its status log.
2. Evaluate any checkpoint with k seeds: see the `do_merge_eval` pattern in any driver (merge FSDP shards → `textgame_eval.sh`/`math_eval.sh` with `--use_bsl true`).
3. Regenerate the three-rule table: `python code/analysis/b4_three_rules.py` over per-env reward JSONs.
4. Check the reproduction fidelity: `cat docs/reproduction_results.md` — every task within 0.3 points of the published OPCD numbers except FrozenLake (+26.3, attributable to batch size / validation argmax / pool draw, analyzed in the run registry).

Driver conventions: all paths are relative to an artifact root (originally `/opt/opcd-artifact`); `say()` lines (`[timestamp] MESSAGE`) are the machine-readable experiment timeline; `RESULT`/`CHAIN`/`COMBO_*` lines are the canonical outcomes cited in the paper.

## Notes

- Logs are verbatim except for (a) trimming of very large files to their final segments and (b) anonymization of usernames/hosts/IPs.
- Checkpoints and raw rollouts are excluded for size; every number in the paper regenerates from the drivers + logs included here.
- The framework directory contains our full patch set on top of the public OPCD release (memory "ten-knife" fixes for 8B consolidation, vllm empty-batch guard, checkpoint-resume wiring).
- The math-8B chain (B6) is the final experiment; its complete results will be in `results/all_results_summary.md` at camera-ready.
