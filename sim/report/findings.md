### 1. Federated learning vs local-only
- **The shared model alone trails a mature site.** Federated balanced accuracy rises slowly (0.769 at 50 events per site to 0.827 at 5,000), while local-only climbs to 0.877. The sites disagree about what is abnormal, and one averaged model cannot hold every site's rule. The cost falls mostly on the back door (0.698 federated vs 0.757 local).
- **Federation wins when data is scarce.** At 50 events per site the federated model beats local-only on balanced accuracy (0.769 vs 0.731) and on anomaly recall (0.544 vs 0.433).
- **Federated + local fine-tune has the best balanced accuracy at every size** (0.771 to 0.879). For anomaly recall it is best from 150 events upward, ties local-only at 1,500, and at 50 events the plain federated model is still better (0.544 vs 0.501). Fine-tuning should wait until a site has a few hundred labelled events.
- **Cold start is the strongest argument for FL.** A camera at a site the fleet has never seen gets 0.752 balanced accuracy and 0.715 anomaly recall on day one. Local-only needs about 150 events to match the accuracy and about 500 to match the recall.
- **Fine-tuning a new camera too early hurts recall.** With 50 or 150 local events, fine-tuning lifts balanced accuracy (0.801, 0.845) but drops anomaly recall below the untouched fleet model (0.557, 0.705 vs 0.715). It only wins on recall from 500 events (0.788).

### 1c. Fine taxonomy and site identity
Slices are small (household car at night: 20 validation events, stranger car at night: 32). Treat slice numbers as direction, not precision.
- The fine taxonomy raises balanced accuracy for every training method: federated +0.03 to +0.045, fine-tuned +0.023 to +0.047, depending on crop-classifier accuracy (76% to 100%).
- **A shared model cannot learn "this is the resident's car."** Federated scores 0.01 to 0.04 on household cars at night with the fine taxonomy. Local fine-tuning restores it to 0.83 to 1.00.
- **Trade-off:** fine-tuning makes a site forget fleet knowledge about rare events. With the 88% classifier, "stranger car at night" falls from 0.89 (federated) to 0.72 after fine-tuning, and "carrier truck out of its window" from 0.96 to 0.83. Fine-tuning needs a guard (fewer epochs, keep the fleet model's anomaly flag as a second opinion, or regularize toward fleet weights).

### 2. Frame rate while training
- **All three Hailo builds held 12 FPS during a real Flower round:** pi5-hailo 12.98, pipelined 24.27, pi5-hailo-fine 13.34. Training a 205K-parameter head on the CPU does not stop detection when the Hailo does the heavy work.
- **Below target:** pi5-threat-on-cpu (10.29 during training, 10.81 before) and the CPU-only fallback (about 2.4).
- **Read the percentages with care.** pi5-hailo went UP 7.5% during training and kept rising in recovery (13.48), which is background load on this PC, not a training effect. Earlier runs measured -8.1% and +0.1%. Which builds pass or fail 12 FPS was consistent across runs; the real numbers will come from a Pi 5.

### 3. Poisoning resistance
- **FedAvg (what the server runs today) is fragile.** One NaN node wipes the fleet model (0.000). A sign-flipped update drops balanced accuracy to 0.356 while recall jumps to 0.978 because the model flags nearly everything, which makes alerts useless. A x6 boosted "hide anomalies" update cuts recall from 0.637 to 0.075.
- **robust.py stops the loud attacks.** It rejected the bad camera in 6 of 6 rounds for NaN, noise, sign flip and the x6 boosted label flip, and rejected 0 of 30 honest updates in every scenario. With no attack it costs nothing (0.859 vs 0.859).
- **Moderate and quiet label flips get through robust.py.** The x2.5 boosted flip was never rejected and pulled recall to 0.449 (0.416 with the validation gate). The unboosted flip lowered recall from 0.64 to 0.56 under every defence.
- **Plain trimmed mean did better on label flips** (0.575 and 0.573 recall on x2.5 and x6) but lets a sign flip skew the model (balanced accuracy 0.772, recall 0.743).
- **The validation gate added nothing here.** It never rejected an update, because it checks balanced accuracy only, and balanced accuracy barely moves under a label flip (0.871 to 0.877).

### Recommendations
1. **Ship federated + per-site fine-tune, start new cameras on the fleet model, and do not fine-tune before about 500 labelled events.** Guard against forgetting rare fleet-learned anomalies.
2. **Wire an aggregation defence into `central_server/fl_aggregator.py` before any multi-site trial.** Consider robust.py's rejection step followed by a trimmed mean instead of a plain mean, since each covers the other's gap in Section 3. This combination is untested.
3. **Make the validation gate check anomaly recall (plus the loss tie-break).** Untested hypothesis: it is the cheapest defence against label flips, since balanced accuracy alone did not move.
4. **Keep the threat model on the Hailo.** The CPU threat build is the only accelerated build under 12 FPS.
5. **Replace the Table 11 budgets with Pi 5 + Hailo measurements** (`scripts/benchmark_classes.py`, then `sim/profiles.py`) and re-run Section 2 on the Pi.
