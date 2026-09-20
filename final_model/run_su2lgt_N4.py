import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from multiseed_validate import run_su2lgt, _save

if __name__ == "__main__":
    print("--- SU(2) LGT qubit-count extension: N=4 (8 qubits) ---")
    final = run_su2lgt(seed=0, num_qubits=8, depth=6, adapt_iters=300, adapt_lr=0.05,
                        pretrain_epochs=40, pretrain_lr=0.01)
    _save("su2_lgt", "adaptation_final_N4.json", final)
