"""Portable offline regressions for the GEPA export-path fix.

Covers the T1 accepted fixes as importable contracts (no model calls,
no network, no private fixtures):
- SkillModule seeds the FULL skill body onto the named predictor's
  mutable signature instructions (GEPA's mutation surface) and forward()
  sends those instructions, never the stale seed attribute;
- the exported body comes from the optimized module's signature via
  extract_optimized_body(), with missing/empty refusing a seed fallback;
- export/reload round-trips a COMPLETE large body containing internal
  Markdown '---' separators, and a fresh module built from the exported
  body receives exactly the exported instructions;
- the fitness metric accepts DSPy 3.4/GEPA's extended call signature
  (pred_name/pred_trace);
- the optimizer builder constructs dspy 3.4 GEPA (max_metric_calls +
  reflection LM); max_steps is gone in dspy 3.4.

Optimizer/LM stubs and dspy's DummyLM are plumbing tests only: nothing
here is a real GEPA run, and no scores are claimed.
"""

import hashlib
from types import SimpleNamespace

import dspy
import pytest

from evolution.skills.skill_module import SkillModule, load_skill, reassemble_skill

BODY = "## Procedure\n1. Analyze the task\n2. Apply the skill\n\n--\nnot a separator\n"

FRONTMATTER = "\nname: portable-skill\ndescription: A portable regression skill\nversion: 1.0.0\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class _StubPredict:
    """Stands in for dspy.Predict: records call kwargs, no LM."""

    def __init__(self, instructions: str):
        self.signature = SimpleNamespace(instructions=instructions)
        self.last_kwargs = None

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(output="stub-output")


class _StubPredictor:
    """Stands in for dspy.ChainOfThought: callable, exposes .predict."""

    def __init__(self, instructions: str):
        self.predict = _StubPredict(instructions)

    def __call__(self, **kwargs):
        return self.predict(**kwargs)


def _stub_optimized_module(instructions=...):
    obj = SimpleNamespace()
    obj.predictor = SimpleNamespace()
    if instructions is ...:
        obj.predictor.predict = None
    else:
        obj.predictor.predict = SimpleNamespace(
            signature=SimpleNamespace(instructions=instructions)
        )
    return obj


# -- SkillModule puts the body on the mutation surface ----------------------


def test_seeded_signature_carries_full_skill_body():
    module = SkillModule(BODY)
    instructions = module.predictor.predict.signature.instructions
    # dspy's with_instructions normalizes trailing whitespace; the seeded
    # surface is the full body modulo that normalization.
    assert instructions == BODY.strip(), "skill body must seed the mutable signature instructions"


def test_forward_sends_signature_instructions_not_stale_seed():
    module = SkillModule("SEED")
    module.predictor = _StubPredictor("EVOLVED")
    out = module(task_input="do the task")
    assert out.output == "stub-output"
    sent = module.predictor.predict.last_kwargs["skill_instructions"]
    assert sent == "EVOLVED"
    assert sent != module.skill_text, "forward() fell back to the stale seed body (T1 bug)"


def test_signature_mutation_is_visible_to_forward():
    module = SkillModule("SEED")
    module.predictor.predict.signature = module.TaskWithSkill.with_instructions("EVOLVED")
    assert module.predictor.predict.signature.instructions == "EVOLVED"
    assert module.skill_text == "SEED", "module attribute must stay stale (mutation surface is the signature)"


# -- optimized-body export ----------------------------------------------------


def test_extract_returns_optimized_signature_instructions():
    from evolution.skills.evolve_skill import extract_optimized_body

    assert extract_optimized_body(_stub_optimized_module("EVOLVED-BODY")) == "EVOLVED-BODY"


@pytest.mark.parametrize(
    "case",
    ["predict_none", "signature_none", "instructions_none", "instructions_empty", "instructions_whitespace"],
)
def test_extract_refuses_missing_or_empty_instructions(case):
    from evolution.skills.evolve_skill import extract_optimized_body

    stubs = {
        "predict_none": _stub_optimized_module(),
        "signature_none": SimpleNamespace(
            predictor=SimpleNamespace(predict=SimpleNamespace(signature=None))
        ),
        "instructions_none": _stub_optimized_module(instructions=None),
        "instructions_empty": _stub_optimized_module(instructions=""),
        "instructions_whitespace": _stub_optimized_module(instructions="  \n  "),
    }
    with pytest.raises(RuntimeError, match="Missing optimized instructions"):
        extract_optimized_body(stubs[case])


def _large_body_with_internal_separators() -> str:
    sections = []
    for i in range(50):
        sections.append(
            f"## Section {i}\n\nDetailed guidance block {i} with enough text to\n"
            f"make the body large: {'word ' * 30}\n"
        )
    return ("\n---\n").join(sections)


def test_export_reload_roundtrip_preserves_complete_large_body(tmp_path):
    body = _large_body_with_internal_separators()
    assert len(body) > 10_000
    assert "\n---\n" in body, "fixture must contain internal Markdown separators"

    original = tmp_path / "SKILL.md"
    original.write_text(f"---{FRONTMATTER}---\n\n{body}\n")
    skill = load_skill(original)
    assert skill["body"] == body.strip()

    exported = tmp_path / "exported.md"
    exported.write_text(reassemble_skill(skill["frontmatter"], skill["body"]))
    reloaded = load_skill(exported)

    assert reloaded["body"] == skill["body"], "complete body must survive export/reload byte-for-byte"
    assert reloaded["body"].count("\n---\n") == body.count("\n---\n")
    assert sha256_text(reloaded["body"]) == sha256_text(skill["body"])


def test_fresh_module_from_exported_body_receives_exactly_exported_instructions(tmp_path):
    body = _large_body_with_internal_separators()
    original = tmp_path / "SKILL.md"
    original.write_text(f"---{FRONTMATTER}---\n\n{body}\n")
    skill = load_skill(original)
    exported = tmp_path / "exported.md"
    exported.write_text(reassemble_skill(skill["frontmatter"], skill["body"]))
    exported_body = load_skill(exported)["body"]

    module = SkillModule(exported_body)
    # dspy normalizes trailing whitespace on signature instructions
    assert module.predictor.predict.signature.instructions == exported_body.strip()
    assert (
        sha256_text(module.predictor.predict.signature.instructions)
        == sha256_text(exported_body.strip())
    )


# -- metric call compatibility --------------------------------------------------


def test_metric_accepts_extended_gepa_call_signature():
    from evolution.core.fitness import skill_fitness_metric

    example = dspy.Example(task_input="summarize", expected_behavior="use concise bullet points").with_inputs(
        "task_input"
    )
    prediction = dspy.Prediction(output="use concise bullet points")
    positional = skill_fitness_metric(example, prediction)
    extended = skill_fitness_metric(
        example, prediction, trace=None, pred_name="predictor.predict", pred_trace=None
    )
    assert positional == extended == 1.0


def test_metric_scores_keyword_overlap_unchanged():
    from evolution.core.fitness import skill_fitness_metric

    example = dspy.Example(task_input="t", expected_behavior="alpha beta gamma").with_inputs("task_input")
    prediction = dspy.Prediction(output="alpha beta")
    score = skill_fitness_metric(example, prediction, pred_name="p", pred_trace=None)
    expected_words = set("alpha beta gamma".split())
    output_words = set("alpha beta".split())
    assert score == pytest.approx(0.3 + 0.7 * (len(expected_words & output_words) / len(expected_words)))


# -- optimizer construction (dspy 3.4 compatibility) ---------------------------


def test_optimizer_builder_constructs_dspy34_gepa():
    """Real object introspection: exact budget and reflection LM identity.

    Budget rule: max(3 * iterations, 12) metric calls. Reflection LM is
    the requested optimizer model at temperature 1.0 / max_tokens 32000.
    """
    from evolution.core.fitness import skill_fitness_metric
    from evolution.skills.evolve_skill import build_gepa_optimizer

    low = build_gepa_optimizer(
        metric=skill_fitness_metric, iterations=4, optimizer_model="openai/gpt-4.1"
    )
    assert low.max_metric_calls == 12, "budget must hit the floor for small iteration counts"

    high = build_gepa_optimizer(
        metric=skill_fitness_metric, iterations=10, optimizer_model="openai/gpt-4.1"
    )
    assert high.max_metric_calls == 30, "budget must be exactly 3x iterations above the floor"

    lm = high.reflection_lm
    assert lm.model == "openai/gpt-4.1"
    assert lm.kwargs["temperature"] == 1.0
    assert lm.kwargs["max_tokens"] == 32000


def test_optimizer_builder_budget_scales_with_iterations():
    from evolution.core.fitness import skill_fitness_metric
    from evolution.skills.evolve_skill import build_gepa_optimizer

    for iterations, expected in ((1, 12), (4, 12), (5, 15), (10, 30), (50, 150)):
        optimizer = build_gepa_optimizer(
            metric=skill_fitness_metric,
            iterations=iterations,
            optimizer_model="openai/gpt-4.1",
        )
        assert optimizer.max_metric_calls == expected == max(3 * iterations, 12)


def test_optimizer_builder_passes_metric_exactly(monkeypatch):
    """Targeted stub capture of the dspy.GEPA constructor call."""
    import evolution.skills.evolve_skill as evolve_skill_module
    from evolution.core.fitness import skill_fitness_metric

    captured = {}

    class _StubGEPA:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(evolve_skill_module.dspy, "GEPA", _StubGEPA)
    evolve_skill_module.build_gepa_optimizer(
        metric=skill_fitness_metric, iterations=10, optimizer_model="openai/gpt-4.1"
    )
    assert captured["metric"] is skill_fitness_metric
    assert captured["max_metric_calls"] == 30
    assert captured["reflection_lm"].model == "openai/gpt-4.1"
    assert "max_steps" not in captured


def test_dspy34_gepa_rejects_max_steps():
    from evolution.core.fitness import skill_fitness_metric

    with pytest.raises(TypeError):
        dspy.GEPA(metric=skill_fitness_metric, max_steps=4)


# -- DummyLM plumbing: a full forward pass offline -------------------------------


def test_forward_with_dummy_lm_plumbing():
    # dspy 3.4 DummyLM takes field dicts; ChainOfThought needs its
    # reasoning field present alongside the output field.
    lm = dspy.utils.DummyLM([{"reasoning": "stub", "output": "plumbing-ok"}])
    module = SkillModule(BODY)
    with dspy.context(lm=lm):
        pred = module(task_input="do the task")
    assert pred.output == "plumbing-ok"
