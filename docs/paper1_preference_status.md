# Paper 1 — Pairwise preference study status

**Study status: BLOCKED**  
**Reason:** real human listeners unavailable in this execution environment.  
**MOS:** **unmeasured** (P.808 not run; UTMOS must not be substituted for MOS).

## Named comparison

| Comparison | Systems | Wav roots (demo5) | Match mode | Self-vs-self traps |
|------------|---------|-------------------|------------|--------------------|
| `silero_frontend_vs_track_a` | Silero+frontend vs Track A | `out/n2_first_execution/silero_frontend/demo5/`, `out/n2_first_execution/track_a/demo5/` | `order` (filenames differ) | 3 |

- **Listener count:** **0**
- **Content comparisons prepared:** 5 (one per demo5 utterance)
- **Trap comparisons prepared:** 3
- **Binomial CI95 on human preferences:** not applicable (no responses)
- **Trap same-rate (panel QC):** not applicable for humans; dry-run self-check shows trap_same_rate=1.0 on synthetic trap-correct answers only

## Systems considered

| System | Included in ballot | Note |
|--------|-------------------|------|
| Silero + shared frontend | yes | fair system-compare condition (R2) |
| Track A (`mel_stress_v01_s4`) | yes | frozen production baseline |
| N2 S1 listen-demo | **no** | N2 gate **FAIL** (`out/n2_gate.json`); wavs exist under `out/n2_first_execution/n2_s1_listen_demo/` for qualitative listen only |

## Runnable protocol (dry-run done)

```bash
export PYTHONPATH=src
python scripts/51_preference_test.py protocol \
  --out out/paper1_figures/preference_protocol.json

python scripts/51_preference_test.py \
  --dir-a out/n2_first_execution/silero_frontend/demo5 \
  --dir-b out/n2_first_execution/track_a/demo5 \
  --name-a silero_frontend --name-b track_a \
  --match-mode order --n-traps 3 --seed 42 \
  --status blocked_no_listeners \
  --out out/paper1_figures/preference_silero_fe_vs_track_a

python scripts/51_preference_test.py self-check \
  --key out/paper1_figures/preference_silero_fe_vs_track_a/key.json \
  --out out/paper1_figures/preference_silero_fe_vs_track_a/self_check.json
```

Artifacts:

- `out/paper1_figures/preference_protocol.json`
- `out/paper1_figures/preference_silero_fe_vs_track_a/{ballot,key,status,pairing,self_check}.json`
- `out/paper1_figures/preference_silero_fe_vs_track_a/ballot.html`

When a human panel is available: collect `{pair_id, choice}` responses, score with
`--score responses.json`, report `listener_count`, `prefer_a`/`prefer_b`,
Wilson CI95, and `trap_same_rate`. Do **not** invent those numbers beforehand.

Machine-readable status also in `out/paper1_figures/preference_protocol.json`
(`study_status: BLOCKED`, `mos_status: unmeasured`).
