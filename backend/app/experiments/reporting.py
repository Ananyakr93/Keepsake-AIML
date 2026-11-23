"""Build the report figures and editable presentation from the same evaluation JSON."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

LABELS = {
    "rule": "Fixed rule", "always_yes": "Always remember", "behavior": "Behavior only", "behavior_text": "Behavior + text",
    "multimodal": "Behavior + text + visual", "vector": "Vector order",
    "weighted": "Fixed feature weights", "logistic": "Logistic reranker",
    "xgboost": "XGBoost, all features", "xgboost_cosine": "XGBoost, cosine only",
    "xgboost_without_term_coverage": "XGBoost, no term coverage",
    "before": "Before feedback", "after": "After feedback",
    "exponential": "Exponential decay", "learned": "Learned interest",
    "random_forest": "Random forest", "intent_logistic": "Logistic regression",
    "intent_xgboost": "XGBoost classifier",
}
NAVY = "#17243a"
TEAL = "#087e80"
SLATE = "#576980"
BACKGROUND = "#f4f7fa"


def _mean(rows, key):
    return rows[key]["mean"]


def _cell(rows, metric):
    if metric not in rows:
        return "—"
    value = rows[metric]
    return f"{value['mean']:.3f} ± {value['fold_sd']:.3f}"


def _table(values, columns):
    headings = "| Method | " + " | ".join(title for title, _ in columns) + " |"
    divider = "|---|" + "---:|" * len(columns)
    rows = [
        "| " + LABELS.get(name, name) + " | "
        + " | ".join(_cell(metrics, key) for _, key in columns) + " |"
        for name, metrics in values.items()
    ]
    return "\n".join([headings, divider, *rows])


def _chart(data, metric, title, path, highlighted=()):
    names = list(data)
    means = [data[name][metric]["mean"] for name in names]
    spreads = [data[name][metric]["fold_sd"] for name in names]
    fig, ax = plt.subplots(figsize=(8.8, max(2.6, len(names) * 0.45 + 1.1)), layout="constrained")
    positions = list(range(len(names)))
    ax.barh(positions, means, xerr=spreads, capsize=3, height=0.6,
            color=[TEAL if name in highlighted else SLATE for name in names])
    ax.set_yticks(positions, [LABELS.get(name, name) for name in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Mean across folds · whiskers show one fold SD")
    ax.set_title(title, loc="left", color=NAVY, weight="bold")
    ax.grid(axis="x", alpha=0.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for position, mean, spread in zip(positions, means, spreads, strict=True):
        ax.text(mean + spread + 0.018, position, f"{mean:.3f}", va="center", fontsize=9)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def _confusion(report, path):
    matrix = np.array(report["intent_confusion"])
    intents = [name.capitalize() for name in report["intent_labels"]]
    fig, ax = plt.subplots(figsize=(7.4, 5.6), layout="constrained")
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(len(intents)), yticks=range(len(intents)),
           xticklabels=intents, yticklabels=intents, xlabel="Predicted intent", ylabel="Actual intent")
    ax.tick_params(axis="x", labelrotation=30)
    ax.set_title("Intent confusion matrix · all held-out sessions", loc="left", color=NAVY, weight="bold")
    for row, column in np.ndindex(matrix.shape):
        count = matrix[row, column]
        ax.text(column, row, str(count), ha="center", va="center",
                color="white" if count > matrix.max() / 2 else NAVY)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def _summary(report):
    ranking = report["ranking"]
    admission = report["admission"]
    feedback = report["feedback"]
    return {
        "intent": _mean(report["intent"], "f1"),
        "rule": _mean(admission["rule"], "f1"),
        "always_yes": _mean(admission["always_yes"], "f1"),
        "intent_stress": _mean(report["intent_stress"], "f1"),
        "behavior": _mean(admission["behavior"], "f1"),
        "text": _mean(admission["behavior_text"], "f1"),
        "visual": _mean(admission["multimodal"], "f1"),
        "vector": _mean(ranking["vector"], "ndcg_at_5"),
        "xgb": _mean(ranking["xgboost"], "ndcg_at_5"),
        "cosine_only": _mean(ranking["xgboost_cosine"], "ndcg_at_5"),
        "rank_no_terms": _mean(ranking["xgboost_without_term_coverage"], "ndcg_at_5"),
        "before": _mean(feedback["before"], "ndcg_at_5"),
        "after": _mean(feedback["after"], "ndcg_at_5"),
        "decay": _mean(report["interests"]["exponential"], "f1"),
        "interest": _mean(report["interests"]["learned"], "f1"),
    }


def _report(report, output):
    n = _summary(report)
    data = report["dataset"]
    latency = report["latency_local_cpu_ms"]
    votes = report["feedback_counts"]
    names = report["fold_count"]
    feedback_folds = sum(
        fold["retrieval"]["feedback"]["after"]["ndcg_at_5"] > fold["retrieval"]["feedback"]["before"]["ndcg_at_5"]
        for fold in report["folds"]
    )
    classification = [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                      ("F1", "f1"), ("ROC-AUC", "roc_auc")]
    rank = [("P@5", "precision_at_5"), ("R@5", "recall_at_5"), ("MRR@5", "mrr_at_5"),
            ("NDCG@5", "ndcg_at_5"), ("Candidate recall@20", "candidate_recall_at_20")]
    report_text = f"""# AIML-EL · Learning What to Remember

## Phase 6 evaluation report

**Evidence source · Synthetic browsing, simulated visual vectors, and scripted relevance.**

AIML-EL uses browsing sessions to predict intent, estimate topic interest,
and decide what to remember. It retrieves up to 20 candidates and reranks
them to five. This benchmark asks how the classifiers, feature families, and
rerankers behave under controlled synthetic conditions.

### What the benchmark found

- Intent classification reached **{n['intent']:.3f} macro F1**. The generator
  gives each intent recognizable cues. Replacing those cues with unseen,
  more ambiguous phrases lowered macro F1 to **{n['intent_stress']:.3f}**.
- Admission F1 was **{n['rule']:.3f}** for the fixed rule, **{n['behavior']:.3f}**
  for behavior, **{n['text']:.3f}** with text, and **{n['visual']:.3f}** with
  visuals added. Always remembering reaches **{n['always_yes']:.3f}** F1 while
  accepting every session. The text variant also uses predicted intent and past topic
  engagement, so its change reflects those features together.
- Full XGBoost reranking reached **{n['xgb']:.3f} NDCG@5** and vector order
  reached **{n['vector']:.3f}**. The cosine-only XGBoost ablation scored
  **{n['cosine_only']:.3f}** and dropping only term coverage gave
  **{n['rank_no_terms']:.3f}**. Extra ranking features reduced performance in
  this synthetic comparison. Scripted feedback changed full-ranker NDCG@5 from **{n['before']:.3f}** to
  **{n['after']:.3f}** and improved {feedback_folds} of {names} test folds.
- The learned interest model reached **{n['interest']:.3f} F1** and the
  exponential-decay baseline reached **{n['decay']:.3f}**.

### Experimental design

- **Dataset** · {data['sessions']} sessions from {data['participants']} fictional
  participants, seven intent classes, and a binary memory-worthiness label.
- **Validation** · {names} outer participant folds. Each participant appears
  in the test split once. Each fold reserves other participants for validation
  and uses the remainder to fit candidates. Validation selects models and
  thresholds. The test partition scores those choices.
- **Leakage controls** · IDs and timestamps are excluded as direct model
  features. Timestamps order completed activity for interest features. Intent
  features used by admission come from out-of-fold training predictions.
  TF-IDF is fitted only on the training prompt and memory text.
- **Ranking protocol** · {report['ranking_queries']} held-out synthetic queries.
  Methods share the same 20-item vector candidate pool and rank the same five
  positions. Feedback is drawn from validation prompts with both relevant and
  irrelevant examples. Ablations refit XGBoost using the indicated columns.
  Each prompt has one relevant memory, so P@5 has a maximum of 0.2 here.
- **Reporting** · Tables show the mean and sample standard deviation across
  folds. Variation across folds describes sensitivity to the participant split.

### Intent classification
Intent precision, recall, and F1 are macro-averaged across the seven classes.

Validation scores for the candidate classifiers

{_table({("intent_" + name if name != "random_forest" else name): score
         for name, score in report['intent_validation'].items()}, classification[:-1])}

Held-out scores for the classifier selected in each fold

{_table({"Selected classifier": report['intent']}, classification)}

![Intent confusion matrix](intent-confusion.png)

### Memory admission and interest

{_table(report['admission'], classification + [("Share admitted", "positive_prediction_rate")])}

![Admission F1 by feature family](admission.png)

{_table(report['interests'], classification)}

### Retrieval and feedback

{_table(report['ranking'], rank)}

![Retrieval benchmark and feature ablations](ranking.png)

{_table(report['feedback'], rank)}

The scripted feedback used {votes['judgments']} judgments, including
{votes['useful']} useful judgments, across {votes['training_query_groups']} query
groups. This is the generator's label mix. User feedback acceptance and memory
correction rate require judgments from real users.

For the local CPU benchmark, vector candidate selection plus full XGBoost
reranking took {latency['median']:.2f} ms median and {latency['p95']:.2f} ms at
the 95th percentile over {latency['queries']} queries. The timing covers the
in-memory TF-IDF experiment. Live application latency includes storage,
embedding, and request overhead.

### Limitations and next evaluation

The synthetic corpus reuses vocabulary and produces visual embeddings from
known rules. The folds separate fictional users who share that generator.
Real intent ambiguity, screenshot understanding, long-term adaptation, and
feedback quality still need a consented and reviewed real-world dataset.
Future evaluation should label query-to-memory relevance independently and
measure context usefulness and correction rate on unseen participants.

### Reproduce and verify

The command in the repository README regenerates this report, its figures,
the presentation, and the fold-level metrics. Dataset seed is
`{report['seed']}` and the input SHA-256 is
`{report['dataset_sha256']}`. See [metrics.json](metrics.json) for dependency
versions, participant assignments, test metrics, and the confusion matrix.

### Presentation practice

1. **Why split by participant?** A model can memorize a participant's habits.
   Holding out the entire participant measures transfer to unseen participants.
2. **What does the visual ablation test?** Compare behavior and text to the
   same feature families with simulated vectors added. Compare the F1 and
   ROC-AUC columns before drawing a conclusion about visual features.
3. **What does feedback show?** The loop changes the fitted reranker. Read the
   held-out before/after metrics to judge the effect of these scripted votes.
4. **What is the next real test?** Collect reviewed sessions and relevance
   judgments, reserve participants for evaluation, and rerun the comparisons.
"""
    (output / "report.md").write_text(report_text, encoding="utf-8")


def _color(value):
    return RGBColor.from_string(value.removeprefix("#").upper())


def _text(slide, value, x, y, w, h, *, size=20, color=NAVY, bold=False):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    text = shape.text_frame
    text.word_wrap = True
    text.margin_left = text.margin_right = Pt(0)
    text.margin_top = Pt(0)
    for index, line in enumerate(value.split("\n")):
        p = text.paragraphs[0] if index == 0 else text.add_paragraph()
        p.text = line
        p.font.name = "Aptos"
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = _color(color)
        p.space_after = Pt(9)
    return shape


def _slide(deck, title, kicker="SYNTHETIC EVALUATION"):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = _color(BACKGROUND)
    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.62), Inches(0.5), Inches(0.42), Inches(0.07))
    accent.fill.solid()
    accent.fill.fore_color.rgb = _color(TEAL)
    accent.line.fill.background()
    _text(slide, kicker, 1.2, 0.39, 10.5, 0.4, size=12, color=TEAL, bold=True)
    _text(slide, title, 0.65, 0.91, 12, 1, size=34, bold=True)
    _text(slide, "AIML-EL · Phases 1–6", 0.65, 7.06, 7, 0.2, size=11, color=SLATE)
    _text(slide, str(len(deck.slides)), 12.1, 7.06, 0.45, 0.22, size=11, color=SLATE)
    return slide


def _notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def _presentation(report, directory):
    data = report["dataset"]
    n = _summary(report)
    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)

    slide = _slide(deck, "Learning What to Remember", "AIML-EL · PROJECT PRESENTATION")
    _text(slide, "Multimodal memory admission and reranking\nfor personalized AI agents",
          0.7, 2.1, 11.8, 1.6, size=32, bold=True)
    _text(slide, "Phases 1–6 · evaluation using fictional browsing sessions", 0.7, 4.65, 11, 0.65,
          size=20, color=TEAL)
    _notes(slide, "Open with the research question. Which activities should an assistant remember, and can a learned reranker recover the right memory when asked?")

    slide = _slide(deck, "From browsing to recalled memory")
    _text(slide, "1–2   Capture, group, review, and export sessions\n"
          "3      Predict intent, model interests, and score admission\n"
          "4      Retrieve 20 candidates, rerank to 5, and accept feedback\n"
          "5      Connect inference to the app and run the synthetic demo\n"
          "6      Compare methods, measure ablations, and report results",
          0.75, 2.0, 11.6, 4.9, size=23)
    _notes(slide, "Explain that the same Phase 2 schema feeds both the synthetic and reviewed data paths. The optional low-disk demo runs with CPU Python packages.")

    slide = _slide(deck, "How the benchmark was split")
    _text(slide, f"{data['sessions']} sessions\n{data['participants']} fictional participants\n{report['fold_count']} participant folds",
          0.75, 2, 6, 3.45, size=29, color=TEAL, bold=True)
    _text(slide, "Train · fit preprocessing and models\n"
          "Validation · choose models, thresholds, and scripted feedback\n"
          "Test · score unseen participants",
          6.35, 2.05, 6.2, 3.2, size=21)
    _notes(slide, "Every participant is held out once. Describe why randomly splitting rows would allow a model to exploit repeated behavior from the same participant.")

    slide = _slide(deck, "Does the feature set change admission?")
    slide.shapes.add_picture(str(directory / "admission.png"), Inches(0.65), Inches(2.05), width=Inches(7.8))
    _text(slide, f"Intent macro F1 · {n['intent']:.3f}\nStress test · {n['intent_stress']:.3f}\n\n"
          f"Behavior + text · {n['text']:.3f} F1\n"
          f"With visual features · {n['visual']:.3f} F1\n\n"
          "Visual vectors were simulated.", 8.7, 2.12, 3.95, 4.5, size=19)
    _notes(slide, "Start with the fixed rule, then add behavioral learning and text-derived context. Compare the last two bars to isolate the added visual feature block.")

    slide = _slide(deck, "Learned reranking and feature ablation")
    slide.shapes.add_picture(str(directory / "ranking.png"), Inches(0.65), Inches(1.95), width=Inches(7.8))
    _text(slide, f"Vector order · {n['vector']:.3f}\nFull XGBoost · {n['xgb']:.3f}\n"
          f"XGBoost cosine only · {n['cosine_only']:.3f}\n"
          f"Without term coverage · {n['rank_no_terms']:.3f}\n\n"
          "All scores are mean NDCG@5.", 8.7, 2.12, 3.95, 4.6, size=19)
    _notes(slide, "The candidate set is fixed across methods. NDCG rewards useful results near the top. Explain why the strongest configuration here used only cosine. The ablation is a diagnostic result under the simulator.")

    slide = _slide(deck, "Interest and feedback comparisons")
    _text(slide, f"Interest F1\n{n['decay']:.3f} exponential decay\n{n['interest']:.3f} learned",
          0.8, 2, 6, 3.8, size=25, color=TEAL, bold=True)
    _text(slide, f"Feedback NDCG@5\n{n['before']:.3f} before scripted votes\n{n['after']:.3f} after scripted votes",
          7.1, 2, 5.5, 3.8, size=25, color=NAVY, bold=True)
    _notes(slide, "Report the direction shown by the figures. The feedback path is operational. Performance is evaluated after refitting on separate validation prompt groups.")

    slide = _slide(deck, "What the synthetic results support")
    _text(slide, "The prototype can classify sessions, admit memories, retrieve context, and learn from votes.\n"
          "Structured intent cues and simulated images shape these scores.\n"
          "Real evaluation needs consented sessions, independent relevance judgments, and held-out participants.",
          0.75, 2.15, 11.4, 4.5, size=24)
    _notes(slide, "Close by discussing the evidence needed for real-world claims. Include intent and memory-worthiness labels, relevance ratings, and enough examples across participants.")

    deck.save(directory / "presentation.pptx")


def write_deliverables(report, output: Path):
    _chart(report["admission"], "f1", "Memory admission F1", output / "admission.png", ("multimodal",))
    _chart(report["ranking"], "ndcg_at_5", "Retrieval NDCG@5", output / "ranking.png", ("xgboost",))
    _confusion(report, output / "intent-confusion.png")
    _report(report, output)
    _presentation(report, output)
