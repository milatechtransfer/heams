from heams_memory_milamem.design.storage_type import (
    BaseCanonicalProjector,
    CanonicalDesignConfig,
    CanonicalDesignShape,
    CanonicalProjectorKind,
    CanonicalMemoryRecord,
    CanonicalMemoryType,
    RawMemoryRecordProjectionInput,
    TypedFlatCanonicalProjector,
    build_canonical_projector,
    canonical_memory_id_for_raw,
    load_canonical_design_config,
    project_raw_memory_record,
)


def test_project_raw_memory_record_yields_one_canonical_record() -> None:
    raw_record = RawMemoryRecordProjectionInput(
        id="raw-1",
        content="Alice likes green tea.",
        metadata={"speaker": "user"},
        sequence=7,
    )

    projected = project_raw_memory_record(raw_record)

    assert isinstance(projected, CanonicalMemoryRecord)
    assert projected.id == canonical_memory_id_for_raw("raw-1")
    assert projected.memory_type is CanonicalMemoryType.OBSERVATION
    assert projected.sequence == 7


def test_project_raw_memory_record_retains_raw_provenance() -> None:
    projected = project_raw_memory_record(
        {
            "id": "raw-22",
            "content": "Bob moved to Montreal.",
            "metadata": {"turn_id": 3},
            "sequence": 22,
        }
    )

    assert projected.raw_source_id == "raw-22"
    assert projected.provenance.raw_record_id == "raw-22"


def test_project_raw_memory_record_preserves_content_and_metadata() -> None:
    raw_record = {
        "id": "raw-3",
        "content": "Carol prefers email updates.",
        "metadata": {"channel": "email", "priority": "low"},
        "sequence": 3,
    }

    projected = project_raw_memory_record(raw_record)

    assert projected.content == raw_record["content"]
    assert projected.metadata == raw_record["metadata"]
    assert projected.metadata is not raw_record["metadata"]


def test_project_raw_memory_record_is_deterministic() -> None:
    raw_record = {
        "id": "raw-5",
        "content": "Dana has a dentist appointment on Friday.",
        "metadata": {"category": "schedule"},
        "sequence": 5,
        "status": "deleted",
        "deleted_at": 123.4,
    }

    first_projection = project_raw_memory_record(raw_record)
    second_projection = project_raw_memory_record(raw_record)

    assert first_projection == second_projection


def test_load_canonical_design_config_from_yaml(tmp_path) -> None:
    config_path = tmp_path / "canonical_design.yaml"
    config_path.write_text(
        "shape: flat\nprojector_kind: typed_flat\nmemory_type: observation\n",
        encoding="utf-8",
    )

    loaded = load_canonical_design_config(config_path)

    assert loaded == CanonicalDesignConfig(
        shape=CanonicalDesignShape.FLAT,
        projector_kind=CanonicalProjectorKind.TYPED_FLAT,
        memory_type=CanonicalMemoryType.OBSERVATION,
    )


def test_build_canonical_projector_returns_typed_flat_projector() -> None:
    projector = build_canonical_projector(
        {"shape": "flat", "projector_kind": "typed_flat"}
    )

    assert isinstance(projector, BaseCanonicalProjector)
    assert isinstance(projector, TypedFlatCanonicalProjector)


def test_typed_flat_projector_projects_raw_record() -> None:
    projector = TypedFlatCanonicalProjector(
        {"shape": "flat", "projector_kind": "typed_flat"}
    )

    projected = projector.project(
        {
            "id": "raw-11",
            "content": "Farah prefers afternoon meetings.",
            "metadata": {"preference": "afternoon"},
            "sequence": 11,
        }
    )

    assert projected.id == "canonical:raw-11"
    assert projected.raw_source_id == "raw-11"


def test_project_raw_memory_record_rejects_graph_shape_for_now() -> None:
    raw_record = {
        "id": "raw-8",
        "content": "Evan reports to Priya.",
        "metadata": {"relation": "reports_to"},
        "sequence": 8,
    }

    try:
        project_raw_memory_record(
            raw_record,
            config={"shape": "graph"},
        )
    except NotImplementedError as exc:
        assert "shape='flat'" in str(exc)
    else:
        raise AssertionError("Expected graph shape to be unsupported for now.")


def test_build_canonical_projector_rejects_llm_kind_for_now() -> None:
    try:
        build_canonical_projector({"projector_kind": "llm"})
    except NotImplementedError as exc:
        assert "projector_kind='typed_flat'" in str(exc)
    else:
        raise AssertionError("Expected llm projector kind to be unsupported for now.")
