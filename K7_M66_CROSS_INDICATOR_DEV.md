# M6.6 Explicit Neighbor × Indicator Interaction (SOURCE-ONLY DEV)

Hypothesis: predicting target indicator A at tile i may benefit not only
from neighbor j's image, but also from **which indicator B** neighbor j's
image is interpreted under.

## Architecture

M6.6 builds the same K=8 label-free physical neighbor set per city as
M6.5, but retains **each neighbor image feature** instead of only using
aggregate statistics. For each (neighbor j, indicator B) combination
it constructs:

`token(j,B) = MLP([project(V_j), project(q_B), product, distance])`

where q_B is a **fixed, label-free Qwen Definition+Relation prototype**
for Population, Food or Education, averaged across 12 source-training
image prompts. No target label or neighbor indicator outcome is used.
The query for target indicator A uses the existing current-tile M6.5
City×Indicator representation. A cross-attention head reads all 8×3
neighbor/indicator tokens and refines the bounded residual student.
Original M6.5 aggregate neighborhood input stays in both arms.

### Two matched training arms

- **cross_indicator**: 3 distinct indicator prototypes at each neighbor,
  allowing target A to select neighbor B.
- **spatial_only**: the *same* 3 source-training prototype representations
  are pooled to one shared indicator-unspecific vector before neighbor
  attention; replicated 3×. Same layers, trainable parameters, image
  features, count of attention slots, frozen anchor, source training
  rows, losses and Qwen tokens.

All 3 indicator prototypes are constructed from SOURCE-train imagery
using only frozen hidden token features. Education target outcomes are
never read. Both arms train separately for fixed 20 epochs, batch256,
with identical initial trainable weights and seed42 shuffled batch
ordering. Evaluation is exclusively source city-held Population/Food
(M6.5's existing dev split). **This does not validate unseen Education
transfer or prove cross-indicator reasoning.** Cross attention could
learn shortcut correlations.

## Execute ONE Python command on an allocated GPU terminal

```bash
git pull --ff-only origin main
python -m unittest discover -s tests \
  -p 'test_k7_m66_cross_indicator.py' -v

python -u scripts/run_k7_m66_cross_indicator_dev.py \
  --full-dir runs/k7/m62_full_seed42 \
  --dev-dir runs/k7/m65_neighborhood_dev_seed42 \
  --neighbor-cache runs/k7/m65_neighbors_k8.pt \
  --graph-cache runs/k7/m66_physical_graph_k8.pt \
  --output runs/k7/m66_cross_indicator_dev_seed42 \
  --epochs 20 --batch-size 256 --device cuda
```

This fits **two student models** (40 model-epochs total), no new Visual
anchor fit, no RemoteCLIP model inference, no Qwen-VL model inference.
Qwen hidden tokens are loaded from the existing read-only SQLite.

Each arm has its own atomic `checkpoint_last.pt` and `metrics.csv`.
Re-run the **same command** after interruption to resume. Completed
arms are automatically skipped. Never run two simultaneous instances
with the same output path. Existing M6.5 Dev/Final models and files
are never overwritten.

Results (automatically written, three decimals):
```bash
cat runs/k7/m66_cross_indicator_dev_seed42/REPORT.md
cat runs/k7/m66_cross_indicator_dev_seed42/SUMMARY.csv
```

Inspect per-source-task differences in MSE, Pearson, mean bias. More
trainable interactions do not necessarily improve out-of-city
generalization. This is only one paired seed and limited held cities;
it is evidence for/against the source-task mechanism, not an official
test result. Because neighborhood visual context may include all
source-CSV unlabeled tiles, the method is **transductive**.
