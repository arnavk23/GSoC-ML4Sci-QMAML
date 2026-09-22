"""
HIGGS classification: seeds 4-7 added to the notebook-01b grid (existing seeds 0-3 in results/higgs_v3/raw_results.json),
taking HIGGS from 4 to 8 seeds (the paper's Limitations ask for 8-10; the vs.-best-classical comparison was
underpowered at p=0.08-0.29). Reproduces research_notebooks/01b cells 1, 3, 5 exactly (same data, split, tasks,
model, hyperparameters). One process per qubit count (argv[1]) with its own results file, resumable per
(qubits, seed, variant). Merge with the existing results afterwards.
"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, torch
from config import config
from higgs_data import load_higgs, train_test_split_df, ALL_FEATURE_COLS
from higgs_tasks import make_tasks
from models import TabularFeatureExtractor, PQCModel, HybridModel, freeze_bn
from meta import outer_loop_qmaml, pretrain_learner_paper, outer_loop_meta_update

NQ = int(sys.argv[1])
RESULTS_DIR = os.path.join(HERE, "results", "higgs_v3_extra")
os.makedirs(RESULTS_DIR, exist_ok=True)
RAW = os.path.join(RESULTS_DIR, f"raw_results_q{NQ}.json")
DATA_PATH = os.path.join(HERE, "data_higgs", "higgs_subsample.csv")

df = load_higgs(180_000, DATA_PATH)
train_df, test_df = train_test_split_df(df, test_frac=0.2)
meta_tasks = make_tasks(train_df, ALL_FEATURE_COLS, "m_bb", bin_count=6, support_size=8, query_size=8, tasks_per_bin=3, seed=42)
test_meta_tasks = make_tasks(test_df, ALL_FEATURE_COLS, "m_bb", bin_count=6, support_size=8, query_size=8, tasks_per_bin=2, seed=43)

DEPTH, EPOCHS = 2, 10
config.Q_DEPTH = DEPTH; config.INNER_STEPS = 5; config.INNER_LR = 0.02
config.OUTER_LR = 5e-3; config.W0_SCALE = 0.01; config.EPOCHS = EPOCHS

def build_model(nq, init_type, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tab = TabularFeatureExtractor(len(ALL_FEATURE_COLS), config.CNN_OUTPUT_DIM, nq)
    X = torch.from_numpy(train_df[ALL_FEATURE_COLS].values.astype(np.float32))
    tab.set_norm_stats(X.mean(0), X.std(0)); freeze_bn(tab)
    pqc_init = "zero" if init_type in ("qmaml_existing", "qmaml_paper") else init_type
    return HybridModel(tab, PQCModel(nq, DEPTH, init_type=pqc_init, bound_angles=True))

res_all = json.load(open(RAW)) if os.path.isfile(RAW) else {}
for seed in [4, 5, 6, 7]:
    for variant in ["zero", "uniform", "gaussian", "qmaml_existing", "qmaml_paper"]:
        key = str((NQ, seed, variant))
        if key in res_all: continue
        config.NUM_QUBITS = NQ
        config.CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "ckpt", f"q{NQ}_s{seed}")
        model = build_model(NQ, variant, seed)
        t0 = time.time()
        if variant == "qmaml_existing":
            r = outer_loop_qmaml(model, meta_tasks, test_meta_tasks, config.OUTER_LR, True, ckpt_name=f"q{NQ}_s{seed}_existing.pth")
        elif variant == "qmaml_paper":
            r = pretrain_learner_paper(model, meta_tasks, test_meta_tasks, config.OUTER_LR, True, ckpt_name=f"q{NQ}_s{seed}_paper.pth")
        else:
            r = outer_loop_meta_update(model, meta_tasks, test_meta_tasks, config.OUTER_LR, True, ckpt_name=f"q{NQ}_s{seed}_{variant}.pth")
        res_all[key] = r
        json.dump(res_all, open(RAW, "w"))
        va = r["val_accuracy"][-1] if r.get("val_accuracy") else None
        print(f"{key} done {time.time()-t0:.0f}s final val_acc={va}", flush=True)
print("HIGGSEXTRA DONE", NQ, flush=True)
