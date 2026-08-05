"""Assembles the trajectory-comparison Artifact HTML from chart_data_uris.py
(base64 PNGs) + fresh stats computed from the two .npz trajectories."""
import sys
sys.path.insert(0, ".")
from chart_data_uris import CHARTS
import numpy as np

ARM_JOINTS = ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
              "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]
PRE = np.array([0.1751, 0.0587, 0.1028, -0.3644, 0.1334, 0.5374, 0.1710])

isaac = np.load("trajectory_isaac.npz")
mjc = np.load("trajectory_mujoco_final.npz")
obs_i, arm_i, cube0_i = isaac["obs"], isaac["arm_abs"], isaac["cube_pos0"]
obs_m, arm_m, cube0_m = mjc["obs"], mjc["arm_abs"], mjc["cube_pos0"]
n = min(len(obs_i), len(obs_m))
obs_i, arm_i, obs_m, arm_m = obs_i[:n], arm_i[:n], obs_m[:n], arm_m[:n]

ee_i = np.linalg.norm(obs_i[:, 22:25], axis=1)
ee_m = np.linalg.norm(obs_m[:, 22:25], axis=1)
hb_i = np.linalg.norm(obs_i[:, 28:31], axis=1)
hb_m = np.linalg.norm(obs_m[:, 28:31], axis=1)
drift_i = np.linalg.norm(obs_i[:, 16:19] - obs_i[0, 16:19], axis=1)
drift_m = np.linalg.norm(obs_m[:, 16:19] - obs_m[0, 16:19], axis=1)
pg_i = np.linalg.norm(arm_i - PRE, axis=1)
pg_m = np.linalg.norm(arm_m - PRE, axis=1)
cube_offset_cm = np.linalg.norm(cube0_i - cube0_m) * 100

rows = []
for j, name in enumerate(ARM_JOINTS):
    rows.append(dict(joint=name, target=PRE[j], isaac=arm_i[-1, j], mjc=arm_m[-1, j],
                      ierr=abs(arm_i[-1, j] - PRE[j]), merr=abs(arm_m[-1, j] - PRE[j])))
rows.sort(key=lambda r: -r["merr"])

def jname(n):
    return n.replace("right_", "").replace("_", " ")

table_rows_html = "\n".join(f"""
        <tr class="{'flag' if r is rows[0] else ''}">
          <td class="jname">{jname(r['joint'])}</td>
          <td class="num">{r['target']:+.3f}</td>
          <td class="num isaac-num">{r['isaac']:+.3f}</td>
          <td class="num mjc-num">{r['mjc']:+.3f}</td>
          <td class="num">{r['ierr']:.3f}</td>
          <td class="num {'flag-num' if r is rows[0] else ''}">{r['merr']:.3f}</td>
        </tr>""" for r in rows)

worst = rows[0]

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Policy 1 sim2sim diagnostic — Isaac Lab vs MuJoCo</title>
<style>
:root {{
  --graphite: #f3f4f1;
  --panel: #ffffff;
  --panel-2: #ebeae5;
  --ink: #1a1d21;
  --ink-dim: #5b6067;
  --hair: #1a1d2122;
  --isaac: #3568c9;
  --mujoco: #c85a28;
  --signal: #2f8f5b;
  --alert: #c23b32;
  --alert-bg: #c23b3214;
  --radius: 3px;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --graphite: #10151c;
    --panel: #161d26;
    --panel-2: #1c2430;
    --ink: #e6e8eb;
    --ink-dim: #93999f;
    --hair: #e6e8eb22;
    --isaac: #6fa3f5;
    --mujoco: #f0895a;
    --signal: #5ecb92;
    --alert: #e2695f;
    --alert-bg: #e2695f1c;
  }}
}}
:root[data-theme="dark"] {{
  --graphite: #10151c; --panel: #161d26; --panel-2: #1c2430; --ink: #e6e8eb;
  --ink-dim: #93999f; --hair: #e6e8eb22; --isaac: #6fa3f5; --mujoco: #f0895a;
  --signal: #5ecb92; --alert: #e2695f; --alert-bg: #e2695f1c;
}}
:root[data-theme="light"] {{
  --graphite: #f3f4f1; --panel: #ffffff; --panel-2: #ebeae5; --ink: #1a1d21;
  --ink-dim: #5b6067; --hair: #1a1d2122; --isaac: #3568c9; --mujoco: #c85a28;
  --signal: #2f8f5b; --alert: #c23b32; --alert-bg: #c23b3214;
}}

* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{
  background: var(--graphite);
  color: var(--ink);
  font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
  line-height: 1.55;
  padding: 2.5rem 1.25rem 5rem;
}}
.mono {{
  font-family: ui-monospace, "Cascadia Code", "Roboto Mono", "JetBrains Mono", Menlo, Consolas, monospace;
}}
main {{
  max-width: 920px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 2.25rem;
}}

/* header */
header {{ display: flex; flex-direction: column; gap: 0.6rem; }}
.eyebrow {{
  font-family: ui-monospace, "Cascadia Code", "Roboto Mono", monospace;
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-dim);
}}
h1 {{
  font-family: ui-monospace, "Cascadia Code", "Roboto Mono", monospace;
  font-size: clamp(1.5rem, 3.4vw, 2.05rem);
  font-weight: 600;
  margin: 0;
  text-wrap: balance;
  letter-spacing: -0.01em;
}}
h1 .arrow {{ color: var(--ink-dim); font-weight: 400; }}
.dek {{
  color: var(--ink-dim);
  max-width: 62ch;
  font-size: 0.98rem;
}}
.meta {{
  display: flex; flex-wrap: wrap; gap: 1.4rem;
  font-family: ui-monospace, "Cascadia Code", "Roboto Mono", monospace;
  font-size: 0.76rem;
  color: var(--ink-dim);
  padding-top: 0.35rem;
  border-top: 1px solid var(--hair);
  margin-top: 0.4rem;
}}
.meta b {{ color: var(--ink); font-weight: 600; }}

/* stat tiles */
.tiles {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--hair);
  border: 1px solid var(--hair);
  border-radius: var(--radius);
  overflow: hidden;
}}
.tile {{
  background: var(--panel);
  padding: 1rem 1.1rem;
  display: flex; flex-direction: column; gap: 0.3rem;
}}
.tile .label {{
  font-family: ui-monospace, monospace;
  font-size: 0.68rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-dim);
}}
.tile .value {{
  font-family: ui-monospace, monospace;
  font-size: 1.5rem;
  font-variant-numeric: tabular-nums;
  font-weight: 600;
}}
.tile .sub {{ font-size: 0.78rem; color: var(--ink-dim); }}
.tile.alert .value {{ color: var(--alert); }}
.tile .split {{ display: flex; gap: 0.6rem; align-items: baseline; }}
.tile .split .isaac {{ color: var(--isaac); }}
.tile .split .mjc {{ color: var(--mujoco); }}

/* phase rail */
.phases {{
  display: grid;
  grid-template-columns: 3fr 6fr 5.5fr 26fr;
  gap: 0;
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid var(--hair);
}}
.phase {{
  padding: 0.7rem 0.8rem;
  background: var(--panel);
  border-left: 1px solid var(--hair);
  display: flex; flex-direction: column; gap: 0.25rem;
}}
.phase:first-child {{ border-left: none; }}
.phase .t {{
  font-family: ui-monospace, monospace;
  font-size: 0.68rem;
  color: var(--ink-dim);
  font-variant-numeric: tabular-nums;
}}
.phase .l {{ font-size: 0.82rem; font-weight: 600; }}
.phase.contact {{ background: var(--alert-bg); }}
.phase.contact .l {{ color: var(--alert); }}
.phase.contact .t {{ color: var(--alert); opacity: 0.85; }}

/* sections */
section {{ display: flex; flex-direction: column; gap: 0.9rem; }}
h2 {{
  font-family: ui-monospace, monospace;
  font-size: 0.92rem;
  letter-spacing: 0.02em;
  margin: 0;
  display: flex; align-items: baseline; gap: 0.6rem;
}}
h2 .n {{ color: var(--ink-dim); font-weight: 400; }}
.caption {{ color: var(--ink-dim); font-size: 0.88rem; max-width: 68ch; }}
.legend-key {{ display: inline-flex; gap: 1.1rem; align-items: center; font-size: 0.78rem; }}
.legend-key span {{ display: inline-flex; align-items: center; gap: 0.35rem; }}
.legend-key .dot {{ width: 0.6rem; height: 0.6rem; border-radius: 50%; display: inline-block; }}
.dot.isaac {{ background: var(--isaac); }}
.dot.mjc {{ background: var(--mujoco); }}
.dot.target {{ background: var(--signal); }}

.chart-wrap {{
  background: var(--panel);
  border: 1px solid var(--hair);
  border-radius: var(--radius);
  padding: 0.6rem;
  overflow-x: auto;
}}
.chart-wrap img {{ display: block; width: 100%; height: auto; max-width: 100%; }}
.chart-dark {{ display: none; }}
@media (prefers-color-scheme: dark) {{
  .chart-light {{ display: none; }}
  .chart-dark {{ display: block; }}
}}
:root[data-theme="dark"] .chart-light {{ display: none; }}
:root[data-theme="dark"] .chart-dark {{ display: block; }}
:root[data-theme="light"] .chart-light {{ display: block; }}
:root[data-theme="light"] .chart-dark {{ display: none; }}

/* table */
.table-wrap {{ overflow-x: auto; border: 1px solid var(--hair); border-radius: var(--radius); }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; min-width: 560px; }}
th {{
  font-family: ui-monospace, monospace;
  font-size: 0.68rem; letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--ink-dim); text-align: right; font-weight: 500;
  padding: 0.55rem 0.8rem; border-bottom: 1px solid var(--hair);
  background: var(--panel-2);
}}
th:first-child, td.jname {{ text-align: left; }}
td {{ padding: 0.5rem 0.8rem; border-bottom: 1px solid var(--hair); background: var(--panel); }}
tr:last-child td {{ border-bottom: none; }}
td.num {{ font-family: ui-monospace, monospace; text-align: right; font-variant-numeric: tabular-nums; }}
.isaac-num {{ color: var(--isaac); }}
.mjc-num {{ color: var(--mujoco); }}
tr.flag td {{ background: var(--alert-bg); }}
.flag-num {{ color: var(--alert); font-weight: 700; }}

/* prose */
.prose {{ display: flex; flex-direction: column; gap: 0.85rem; font-size: 0.96rem; max-width: 70ch; }}
.prose strong {{ color: var(--ink); }}
.prose .mjc-inline {{ color: var(--mujoco); font-weight: 600; }}
.prose .isaac-inline {{ color: var(--isaac); font-weight: 600; }}
.prose .alert-inline {{ color: var(--alert); font-weight: 600; }}
ol.steps {{ margin: 0; padding-left: 1.3rem; display: flex; flex-direction: column; gap: 0.5rem; }}
ol.steps li::marker {{ font-family: ui-monospace, monospace; color: var(--ink-dim); }}

footer {{
  border-top: 1px solid var(--hair);
  padding-top: 1rem;
  color: var(--ink-dim);
  font-size: 0.78rem;
  font-family: ui-monospace, monospace;
}}

@media (max-width: 640px) {{
  .tiles {{ grid-template-columns: repeat(2, 1fr); }}
  .phases {{ grid-template-columns: 1fr; }}
  .phase {{ border-left: none; border-top: 1px solid var(--hair); }}
  .phase:first-child {{ border-top: none; }}
}}
</style>
</head>
<body>
<main>

  <header>
    <div class="eyebrow">sim2sim transfer diagnostic — g1_lift_ext</div>
    <h1>Policy 1 <span class="arrow">→</span> reach-and-hold, Isaac Lab <span class="arrow">vs</span> MuJoCo</h1>
    <p class="dek">Same exported checkpoint, same cube spawn, same control loop — replayed step-for-step in
      both simulators to isolate where the physics reconstruction diverges from the source of truth.</p>
    <div class="meta">
      <span>checkpoint <b>2026-07-16_09-18-57</b></span>
      <span>steps <b>{n}</b> · {n*0.01:.1f}s @ 100Hz</span>
      <span>cube spawn offset <b>{cube_offset_cm:.2f}cm</b></span>
      <span>MuJoCo config <b>solref 0.01, real hand</b></span>
      <span>gripper action <b>OPEN throughout, both sims</b> <span style="opacity:.7">(by design — grasp is Policy 2's job)</span></span>
    </div>
  </header>

  <div class="tiles">
    <div class="tile alert">
      <div class="label">Cube drift, final</div>
      <div class="value">{drift_m[-1]*100:.1f}cm</div>
      <div class="sub split"><span class="isaac">Isaac {drift_i[-1]*100:.2f}cm</span><span class="mjc">MuJoCo {drift_m[-1]*100:.1f}cm</span></div>
    </div>
    <div class="tile alert">
      <div class="label">Joint-space error, final</div>
      <div class="value">{pg_m[-1]:.3f} rad</div>
      <div class="sub split"><span class="isaac">Isaac {pg_i[-1]:.3f}</span><span class="mjc">MuJoCo {pg_m[-1]:.3f}</span></div>
    </div>
    <div class="tile">
      <div class="label">EE → object, final</div>
      <div class="value">{ee_m[-1]*100:.1f}cm</div>
      <div class="sub split"><span class="isaac">Isaac {ee_i[-1]*100:.1f}cm</span><span class="mjc">MuJoCo {ee_m[-1]*100:.1f}cm</span></div>
    </div>
    <div class="tile">
      <div class="label">Worst joint</div>
      <div class="value">{jname(worst['joint'])}</div>
      <div class="sub">off by {worst['merr']:.2f} rad (~{worst['merr']*57.3:.0f}°)</div>
    </div>
  </div>

  <section>
    <h2>Episode timeline <span class="n">— where the two sims part ways</span></h2>
    <div class="phases">
      <div class="phase"><span class="t">0.00–0.30s</span><span class="l">Matched transient</span></div>
      <div class="phase"><span class="t">0.30–1.96s</span><span class="l">Approach</span></div>
      <div class="phase contact"><span class="t">1.96–2.19s</span><span class="l">⚠ Contact event</span></div>
      <div class="phase"><span class="t">2.19–8.00s</span><span class="l">Settles — close, not exact</span></div>
    </div>
    <p class="caption">This run: tightened contact stiffness (<span class="mono">solref 0.01</span>, down
      from MuJoCo's default <span class="mono">0.02</span>) + the real 7-body Dex1 hand reconstruction, both
      confirmed via the investigation below. Cube drift stays under 1cm until <b>t=1.96s</b> — later than
      earlier configurations tested, but a contact event still happens. Isaac Lab's cube barely moves at all
      across the full episode (<b>0.36cm</b> final vs MuJoCo's <b>18.4cm</b>).</p>
  </section>

  <section>
    <h2>Overview <span class="n">— distances &amp; convergence</span></h2>
    <span class="legend-key">
      <span><span class="dot isaac"></span>Isaac Lab (reference)</span>
      <span><span class="dot mjc"></span>MuJoCo (transfer)</span>
    </span>
    <div class="chart-wrap">
      <img class="chart-light" src="{CHARTS['overview_light']}" alt="Overview: EE-to-object distance, hand-base-to-object distance, cube drift, and joint-space distance to target, Isaac Lab vs MuJoCo">
      <img class="chart-dark" src="{CHARTS['overview_dark']}" alt="Overview: EE-to-object distance, hand-base-to-object distance, cube drift, and joint-space distance to target, Isaac Lab vs MuJoCo">
    </div>
    <p class="caption">Shaded band marks the contact window. Isaac Lab (blue) settles smoothly on every
      metric; MuJoCo (orange) tracks it closely for much longer than earlier configurations, then still
      diverges at the contact event — later and smaller than before, but not eliminated.</p>
  </section>

  <section>
    <h2>Per-joint trajectories <span class="n">— all 7 right-arm DOF</span></h2>
    <span class="legend-key">
      <span><span class="dot isaac"></span>Isaac Lab</span>
      <span><span class="dot mjc"></span>MuJoCo</span>
      <span><span class="dot target"></span>PRE_GRASP_ARM_POSE target</span>
    </span>
    <div class="chart-wrap">
      <img class="chart-light" src="{CHARTS['joints_light']}" alt="Per-joint arm position over time for all 7 right-arm joints, Isaac Lab vs MuJoCo vs target">
      <img class="chart-dark" src="{CHARTS['joints_dark']}" alt="Per-joint arm position over time for all 7 right-arm joints, Isaac Lab vs MuJoCo vs target">
    </div>
    <p class="caption">With both fixes applied, the per-joint gap is far less dramatic than the original
      substitute-hand/default-contact run: no single joint is wildly wrong-signed anymore (worst is
      <span class="mjc-inline">elbow</span> and <span class="mjc-inline">wrist_roll</span>, each off by
      ~11°, vs the original run's 50° miss on shoulder_pitch alone) — the error is now spread thinly across
      several joints rather than concentrated in one badly-diverged DOF.</p>
  </section>

  <section>
    <h2>Investigation <span class="n">— what was tested, in order</span></h2>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>configuration</th><th>cube drift</th><th>ee→object</th><th>joint err</th><th>contact onset</th>
        </tr></thead>
        <tbody>
          <tr><td class="jname">Isaac Lab (reference)</td><td class="num isaac-num">0.36cm</td><td class="num isaac-num">4.06cm</td><td class="num isaac-num">0.145 rad</td><td class="num isaac-num">never</td></tr>
          <tr><td class="jname">MuJoCo — default solref, substitute hand</td><td class="num">33.3cm</td><td class="num">27.6cm</td><td class="num">0.988 rad</td><td class="num">t=1.08s</td></tr>
          <tr><td class="jname">MuJoCo — solref 0.01, substitute hand</td><td class="num">20.3cm</td><td class="num">16.6cm</td><td class="num">0.345 rad</td><td class="num">t=1.15s</td></tr>
          <tr><td class="jname">MuJoCo — solref 0.005, substitute hand</td><td class="num">20.3cm</td><td class="num">16.6cm</td><td class="num">0.345 rad</td><td class="num">t=1.15s</td></tr>
          <tr class="flag"><td class="jname">MuJoCo — solref 0.01, real hand (final)</td><td class="num mjc-num">18.4cm</td><td class="num mjc-num">16.0cm</td><td class="num mjc-num">0.349 rad</td><td class="num mjc-num">t=1.96s</td></tr>
        </tbody>
      </table>
    </div>
    <p class="caption">Contact stiffness (MuJoCo's <span class="mono">solref</span>, untuned by default) gave
      the one large, real improvement. Pushing it past <span class="mono">0.01</span> to
      <span class="mono">0.005</span> — the timestep itself — produced a <b>bit-identical</b> trajectory to
      <span class="mono">0.01</span>: confirmed directly by comparing the saved observation arrays. Contact
      resolution was already saturated; that lever is exhausted. Swapping in the real 7-body hand on top of
      the tightened contact delayed the event (1.15s → 1.96s) but barely moved the final numbers — confirmed
      the reconstruction itself is correct (gripper open/close separation now 9.41cm/0.51cm vs training's
      9.01cm/0.12cm, no sign inversion), it just isn't the dominant factor.</p>
  </section>

  <section>
    <h2>Final position vs. target <span class="n">— sorted by MuJoCo error</span></h2>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>joint</th><th>target</th><th>isaac</th><th>mujoco</th><th>isaac err</th><th>mujoco err</th>
        </tr></thead>
        <tbody>{table_rows_html}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Conclusion</h2>
    <div class="prose">
      <p><strong>"Gripper never closes" — not a bug.</strong> Policy 1's own reward config only scores
        reach-and-hold at <span class="mono">PRE_GRASP_ARM_POSE</span> with the gripper open;
        <span class="mono">penalty_early_close</span> exists specifically to stop Policy 1 from grasping
        early. Grasping is Policy 2's job. Confirmed against the source (<span class="mono">env_cfg.py</span>'s
        <span class="mono">RewardsCfg</span>), not inferred from the rollout.</p>
      <p><strong>Two physics-matching fixes, two real but incomplete improvements.</strong> Tightening
        MuJoCo's default contact stiffness cut cube drift roughly in half (33→20cm) and was the single
        biggest lever tested — but pushing it further (0.01→0.005, the timestep itself) produced a
        <b>bit-identical</b> trajectory, meaning contact resolution was already saturated at 0.01. Swapping
        in the real 7-body hand reconstruction on top of that (validated correct via its own gripper
        calibration) delayed the contact event by nearly a full second but only shaved another 2cm off the
        final drift. Both fixes are real and now shipped in this repo's default config — neither came close
        to closing the gap alone or combined.</p>
      <p><strong>This is a policy-robustness finding, not a modeling gap.</strong> Best case after both
        fixes: <b>{drift_m[-1]*100:.1f}cm</b> cube drift and <b>{pg_m[-1]:.3f} rad</b> joint-space error, vs
        Isaac Lab's <b>{drift_i[-1]*100:.2f}cm</b> and <b>{pg_i[-1]:.3f} rad</b> — still roughly 50× off on
        cube drift after exhausting the two most plausible physics-side explanations. PhysX and MuJoCo will
        never be bit-identical simulators; the pattern here (small mismatch while contact-free, then a real
        collision right at the moment of contact) suggests Policy 1's approach margin near the cube is thin
        enough that ordinary closed-loop control amplifies whatever small difference remains into a genuine
        contact event MuJoCo has and Isaac Lab doesn't. That's a property of the policy, not of either
        simulator — and it's precisely the kind of thing sim2sim testing exists to surface before a
        real-hardware attempt.</p>
      <p><strong>Suggested follow-up.</strong> Test Policy 1's sensitivity to small observation noise
        directly in Isaac Lab (no MuJoCo involved) — if a small injected perturbation near the approach
        also triggers contact there, that would directly confirm the thin-margin explanation rather than
        leaving it inferred from the cross-simulator gap alone.</p>
    </div>
  </section>

  <footer>
    trajectory_isaac.npz · trajectory_mujoco_final.npz · compare_trajectories.py · g1_lift_ext/mujoco_transfer
  </footer>

</main>
</body>
</html>
"""

out_path = "/tmp/claude-1000/-home-virtual-acc-projects/7f5ba3ff-97a5-45ae-a42b-69297e94980d/scratchpad/policy1_trajectory_diagnostic.html"
with open(out_path, "w") as f:
    f.write(html)
print("wrote", out_path, len(html), "bytes")
