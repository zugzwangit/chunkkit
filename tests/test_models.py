from __future__ import annotations

import pytest
from pydantic import ValidationError

from chunkkit import Acl, AclResolution, ModelTarget, Principal, SourceSpan, Visibility


def test_model_target_computes_available_budget() -> None:
    target = ModelTarget(max_input_tokens=1_000, reserved_tokens=100, safety_margin_tokens=25)
    assert target.available_tokens == 875


def test_model_target_rejects_exhausted_budget() -> None:
    with pytest.raises(ValidationError, match="exhaust"):
        ModelTarget(max_input_tokens=100, reserved_tokens=100, safety_margin_tokens=0)


def test_restricted_acl_needs_principal_when_complete() -> None:
    with pytest.raises(ValidationError, match="allowed principal"):
        Acl(visibility=Visibility.RESTRICTED)
    acl = Acl(
        visibility=Visibility.RESTRICTED,
        allow=(Principal(kind="group", identifier="engineering"),),
    )
    assert acl.allow[0].key == "group:engineering"


def test_incomplete_acl_can_represent_unresolved_policy() -> None:
    acl = Acl(visibility=Visibility.RESTRICTED, resolution=AclResolution.INCOMPLETE)
    assert acl.allow == ()


def test_source_span_validation() -> None:
    assert SourceSpan(start=0, end=0).start == 0
    assert SourceSpan(pointer="/body/0").pointer == "/body/0"
    with pytest.raises(ValidationError):
        SourceSpan(start=10, end=2)
