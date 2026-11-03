# K6.2-A: Strict Nested-Pseudo-Task Readiness Audit

This is a **read-only protocol audit**, not model training, hyperparameter
search, or a claim of improved reliability routing. It checks whether
previously trained K5-A nested source-only experts have the correct
pseudo-unseen-task and city isolation for future regional reliability work.

For each of five inner source pseudo-targets, it verifies:
- Nested `last.pt` lists exactly four experts and excludes its pseudo-target;
- The entire nested dataset matches **exactly** the outer K4 train rows,
  rather than outer meta-UCUI cities or any true unseen target indicators;
- Inner train/SCUI/UCSI/UCUI index sets are mutually disjoint;
- Pseudo-target examples occur only in the inner SCUI and UCUI partitions;
- Inner SCUI cities are train-visible, inner UCUI cities are held out;
- Hidden dimensions are recorded for inspection, **not** used to justify
  transferring a trained regional router across incompatible banks.

## Run: existing three K5-A folders, no retraining

```bash
git pull origin main
python -m py_compile scripts/audit_ktrans_k62a.py
python -m unittest discover -s tests -p 'test_ktrans_k62a.py' -v

for task in education nighttime digital_total_cell; do
  python -u scripts/audit_ktrans_k62a.py \
    --k5a-dir "runs/k5a_${task}"
done
```

Each folder gets `k62a_readiness/fold_audit.csv` and
`k62a_readiness/readiness.json`. The first should contain five
rows, one per inner pseudo-target; each terminal report should say
`row_isolation=PASS` for each row.

The audit reports `reliability_supervision_ready=false` **by design**:
K5-A nested models give genuinely pseudo-target-excluded source expert
predictions, but did not train an independent source-only Region Router
for each nested fold. Reusing the outer K5-B2 Region Router directly is
methodologically invalid because it is fitted against a different
expert feature representation and source-expert set. Equal input or
hidden dimensions do not establish representation alignment.

## Research significance

A valid future K6.2-B comparison requires regional predictions from a
regional router trained only on the inner four source tasks, alongside
global K5-A routing constructed with the same inner expert bank.
Calculate regional-minus-global error on heldout inner pseudo-target
labels strictly as *crossfit supervision* in an appropriate training fold.
Labels from the outer heldout task and final target indicators must
remain entirely inaccessible to reliability-model optimization and
selection. K4 outer `best.pt` remains meta-selected; all reported
results until a separate strict evaluation are DEV only.

A passed K6.2-A audit proves data provenance and that nested banks
exist; it does NOT prove regional improvements or reliability
supervision readiness.
