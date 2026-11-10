import pytest
from pydantic import BaseModel

from assistant.errors import BadModelOutput
from assistant.llm import LLMClient, Params
from assistant.structured import ask_structured, extract_json


class Item(BaseModel):
    name: str
    value: int


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('Sure! Here it is:\n```json\n{"a": 1}\n```\nHope it helps.', {"a": 1}),
    ('{"a": [1, 2,], }', {"a": [1, 2]}),                       # trailing commas
    ('reasoning... <answer>{"a": 2}</answer>', {"a": 2}),      # chain-of-thought
    ('[{"a": 1}]', [{"a": 1}]),
    ('{"a": 1, // comment\n "b": 2}', {"a": 1, "b": 2}),
])
def test_extract_json(text, expected):
    assert extract_json(text) == expected


@pytest.mark.parametrize("text", ["no json here", '{"a": [1, 2', '{"a": nope}'])
def test_extract_json_errors(text):
    with pytest.raises(ValueError):
        extract_json(text)


def test_repair_loop_fixes_invalid_output():
    c = LLMClient("demo", demo_answers={"step": ['{"name": "x", "value": "many"}'],
                                        "step:repair": ['{"name": "x", "value": 3}']})
    item = ask_structured(c, task="step", technique="t", system="s", user="u", schema=Item, params=Params())
    assert item.value == 3
    assert [r.task for r in c.log] == ["step", "step:repair"]
    assert "value" in c.log[1].user                     # the validator error is sent back to the model


def test_repair_gives_up():
    c = LLMClient("demo", demo_answers={"step": ["nothing"], "step:repair": ["still nothing"]})
    with pytest.raises(BadModelOutput):
        ask_structured(c, task="step", technique="t", system="s", user="u", schema=Item, params=Params(), repairs=2)
    assert len(c.log) == 3
