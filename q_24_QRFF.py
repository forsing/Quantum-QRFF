"""
QRFF - Quantum Random Fourier Features
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
NUM_FOURIER = 8
LAMBDA_REG = 0.01


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def value_to_features(v):
    theta = v * np.pi / 31.0
    return np.array([theta * (k + 1) for k in range(NUM_QUBITS)])


def quantum_fourier_feature(x, omega, bias):
    qc = QuantumCircuit(NUM_QUBITS)

    projected = float(np.dot(x, omega))

    for i in range(NUM_QUBITS):
        qc.ry(projected * (i + 1) + bias, i)

    for i in range(NUM_QUBITS - 1):
        qc.cx(i, i + 1)

    for i in range(NUM_QUBITS):
        qc.rz(projected * 0.5 + bias * (i + 1), i)

    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities()

    cos_feat = sum(p * np.cos(np.pi * idx / 31)
                   for idx, p in enumerate(probs))
    sin_feat = sum(p * np.sin(np.pi * idx / 31)
                   for idx, p in enumerate(probs))

    return cos_feat, sin_feat


def build_rff_features(X_feats):
    rng = np.random.RandomState(SEED)
    omegas = rng.randn(NUM_FOURIER, NUM_QUBITS) * 2.0
    biases = rng.uniform(0, 2 * np.pi, NUM_FOURIER)

    n = len(X_feats)
    feat_dim = NUM_FOURIER * 2
    F = np.zeros((n, feat_dim))

    for i, x in enumerate(X_feats):
        for j in range(NUM_FOURIER):
            c, s = quantum_fourier_feature(x, omegas[j], biases[j])
            F[i, 2 * j] = c
            F[i, 2 * j + 1] = s

    F /= np.sqrt(NUM_FOURIER)
    return F


def ridge_fit_predict(X, y, lam=LAMBDA_REG):
    alpha = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    return X @ alpha


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    n_states = 1 << NUM_QUBITS
    X_feats = np.array([value_to_features(v) for v in range(n_states)])

    print(f"\n--- Quantum Random Fourier Features ({NUM_QUBITS}q, "
          f"{NUM_FOURIER} frekvencija) ---")
    print(f"  Generisanje QRFF...", end=" ", flush=True)
    F = build_rff_features(X_feats)
    print(f"{F.shape[0]}x{F.shape[1]}")

    print(f"\n--- QRFF regresija po pozicijama ---")
    dists = []
    for pos in range(7):
        y = build_empirical(draws, pos)
        pred = ridge_fit_predict(F, y)
        pred = pred - pred.min()
        if pred.sum() > 0:
            pred /= pred.sum()
        dists.append(pred)

        top_idx = np.argsort(pred)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{pred[i]:.3f}" for i in top_idx)
        print(f"  Poz {pos+1} [{MIN_VAL[pos]}-{MAX_VAL[pos]}]: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QRFF, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()


"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- Quantum Random Fourier Features (5q, 8 frekvencija) ---
  Generisanje QRFF... 32x16

--- QRFF regresija po pozicijama ---
  Poz 1 [1-33]: 2:0.137 | 3:0.112 | 1:0.102
  Poz 2 [2-34]: 7:0.058 | 5:0.056 | 4:0.053
  Poz 3 [3-35]: 16:0.058 | 19:0.053 | 17:0.052
  Poz 4 [4-36]: 20:0.055 | 18:0.055 | 19:0.050
  Poz 5 [5-37]: 29:0.052 | 24:0.050 | 21:0.048
  Poz 6 [6-38]: 33:0.068 | 32:0.064 | 34:0.063
  Poz 7 [7-39]: 38:0.078 | 34:0.063 | 7:0.060

==================================================
Predikcija (QRFF, deterministicki, seed=39):
[2, 7, x, y, z, 33, 38]
==================================================
"""



"""
QRFF - Quantum Random Fourier Features

8 nasumicnih frekvencija (omega vektori + bias) iz seeda
Za svaku frekvenciju: kvantno kolo enkodira projekciju x . omega u Ry+CX+Rz strukturu
Iz Born distribucije izvlaci cos i sin feature - kvantna aproksimacija Fourieovog kernela
Feature vektor: 8 frekvencija x 2 (cos+sin) = 16 dimenzija, normalizovano sa sqrt(8)
Teorijska osnova: Random Fourier Features aproksimiraju kernel funkciju u beskonacnom prostoru
Kvantno kolo zamenjuje klasicnu cos/sin evaluaciju - nelinearna kvantna projekcija
Ridge regresija, deterministicki, brz
"""
