"""
FiRA Judgement Aggregation Starter Code
Dosen: Zico Pratama Putra
Kelompok: [Isi nama anggota kelompok]
"""

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def load_raw_judgements(file_path: str) -> pd.DataFrame:
    """Load raw FiRA judgements"""
    df = pd.read_csv(file_path, sep="\t")
    print(f"Loaded {len(df)} raw judgements")
    print("Columns:", df.columns.tolist())
    print(df.head())
    return df


def simple_majority_vote(group):
    """Baseline: Simple majority vote"""
    votes = group["judgement"].value_counts()
    return votes.idxmax()


def compute_annotator_reliability(df):
    maj = (
        df.groupby(["query_id", "doc_id"])
        .apply(simple_majority_vote)
        .reset_index(name="majority")
    )
    merged = df.merge(maj, on=["query_id", "doc_id"])
    merged["agree"] = merged["judgement"] == merged["majority"]
    reliability = merged.groupby("annotator_id")["agree"].mean().to_dict()
    return reliability


def advanced_aggregation(
    group, strategy="confidence_weighted", annotator_reliability=None
):
    judgements = group["judgement"].astype(float)
    std_score = np.std(judgements, ddof=0)

    # Gate: disagreement tinggi → median (safety net)
    if len(judgements) >= 3 and std_score > 1.0:
        final_score = judgements.median()
    else:
        confidences = group["confidence"].astype(float)
        if strategy == "confidence_weighted":
            # Point 1: Weighted voting berdasarkan confidence annotator
            final_score = np.average(judgements, weights=confidences)
        elif strategy == "annotator_reliability" and annotator_reliability is not None:
            # Point 2: Weighted voting berdasarkan reliability annotator
            weights = group["annotator_id"].map(annotator_reliability).fillna(0.5)
            final_score = np.average(judgements, weights=weights)
        else:
            final_score = judgements.mean()

    return round(final_score)


def aggregate_judgements(
    df: pd.DataFrame,
    method="advanced",
    strategy="confidence_weighted",
    annotator_reliability=None,
) -> pd.DataFrame:
    """Main aggregation function"""
    grouped = df.groupby(["query_id", "doc_id"])

    aggregated = []
    for (qid, did), group in grouped:
        if method == "majority":
            score = simple_majority_vote(group)
            agreement = group["confidence"].mean()
        else:
            score = advanced_aggregation(
                group, strategy=strategy, annotator_reliability=annotator_reliability
            )
            agreement = group["confidence"].mean()

        aggregated.append(
            {
                "query_id": int(qid),
                "doc_id": str(did),
                "score": int(score),
                "num_judgements": len(group),
                "std_score": float(np.std(group['judgement'])),
            }
        )

    result_df = pd.DataFrame(aggregated)
    print(f"Aggregated into {len(result_df)} unique query-doc pairs")
    return result_df


def save_qrels(aggregated_df: pd.DataFrame, output_path: str):
    """Save in TREC qrels format"""
    with open(output_path, "w") as f:
        for _, row in aggregated_df.iterrows():
            f.write(f"{row['query_id']} 0 {row['doc_id']} {row['score']}\n")
    print(f"Qrels saved to {output_path}")


# ====================== MAIN ======================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = load_raw_judgements(os.path.join(script_dir, "../data/fira_raw_judgements.tsv"))

    # Baseline: majority vote
    agg_maj = aggregate_judgements(df, method="majority")

    # Advanced: confidence_weighted (point 1)
    agg_weighted = aggregate_judgements(
        df, method="advanced", strategy="confidence_weighted"
    )

    # Advanced: annotator_reliability (point 2)
    reliability = compute_annotator_reliability(df)
    print(f"\nAnnotator Reliability: {reliability}")
    agg_reliability = aggregate_judgements(
        df,
        method="advanced",
        strategy="annotator_reliability",
        annotator_reliability=reliability,
    )

    # Simpan qrels dari setiap strategi
    data_dir = os.path.join(script_dir, "../data")
    save_qrels(agg_maj, os.path.join(data_dir, "fira_aggregated.qrels"))
    save_qrels(agg_weighted, os.path.join(data_dir, "fira_aggregated_confidence_weighted.qrels"))
    save_qrels(agg_reliability, os.path.join(data_dir, "fira_aggregated_annotator_reliability.qrels"))

    print("\n=== PERBANDINGAN BASELINE (MAJORITY) VS ADVANCED ===")
    print(f"{'Method':<30} {'Score Dist':<30} {'Mean Std Score':<15}")
    print("-" * 75)
    for name, result in [
        ("majority (baseline)", agg_maj),
        ("confidence_weighted", agg_weighted),
        ("annotator_reliability", agg_reliability),
    ]:
        dist = result["score"].value_counts().sort_index().to_dict()
        mean_std = result["std_score"].mean()
        print(f"{name:<30} {str(dist):<30} {mean_std:<15.3f}")

    print("\n=== ANALISIS MANUAL: Contoh Query-Doc Pair ===")
    sample_pairs = [(1, "d1_3"), (1, "d1_8"), (2, "d2_1")]
    for qid, did in sample_pairs:
        raw = df[(df["query_id"] == qid) & (df["doc_id"] == did)]
        print(f"\nQuery {qid}, Doc {did}:")
        for _, row in raw.iterrows():
            rel = reliability.get(row["annotator_id"], 0)
            print(
                f"  {row['annotator_id']}: judgement={row['judgement']}, confidence={row['confidence']:.3f}, reliability={rel:.3f}"
            )
        maj_score = agg_maj[(agg_maj["query_id"] == qid) & (agg_maj["doc_id"] == did)][
            "score"
        ].values[0]
        w_score = agg_weighted[
            (agg_weighted["query_id"] == qid) & (agg_weighted["doc_id"] == did)
        ]["score"].values[0]
        r_score = agg_reliability[
            (agg_reliability["query_id"] == qid) & (agg_reliability["doc_id"] == did)
        ]["score"].values[0]
        print(
            f"  -> majority={maj_score}, confidence_weighted={w_score}, annotator_reliability={r_score}"
        )
