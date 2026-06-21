import pytest
from pydantic import ValidationError

from core.schema import ActionType, Plan, Step


def test_step_accepts_valid_action():
    step = Step(action="llm", input="hi")
    assert step.action == ActionType.LLM


def test_step_rejects_unknown_action():
    with pytest.raises(ValidationError):
        Step(action="delete_everything", input="x")


def test_step_rejects_extra_fields():
    with pytest.raises(ValidationError):
        Step(action="llm", input="x", bogus="y")


def test_step_coerces_none_input_to_empty_string():
    step = Step(action="llm", input=None)
    assert step.input == ""


def test_plan_rejects_too_many_steps():
    with pytest.raises(ValidationError):
        Plan(steps=[{"action": "llm", "input": str(i)} for i in range(20)])


def test_plan_clears_out_of_range_depends_on():
    plan = Plan(steps=[{"action": "llm", "input": "a", "depends_on": 5}])
    assert plan.steps[0].depends_on is None


def test_plan_clears_forward_reference_depends_on():
    plan = Plan(
        steps=[
            {"action": "llm", "input": "a", "depends_on": 1},
            {"action": "llm", "input": "b"},
        ]
    )
    assert plan.steps[0].depends_on is None


def test_plan_keeps_valid_depends_on():
    plan = Plan(
        steps=[
            {"action": "llm", "input": "a"},
            {"action": "summarize", "input": "{{result}}", "depends_on": 0},
        ]
    )
    assert plan.steps[1].depends_on == 0
