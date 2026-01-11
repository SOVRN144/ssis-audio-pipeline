"""Test that all JSON schemas in specs/ are valid Draft 2020-12 schemas."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for

SPECS_DIR = Path(__file__).parent.parent / "specs"
EXPECTED_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"


def discover_schema_files() -> list[Path]:
    """Discover all schema files in the specs directory."""
    return sorted(SPECS_DIR.glob("*.schema.json"))


@pytest.mark.parametrize(
    "schema_path",
    discover_schema_files(),
    ids=lambda p: p.name,
)
def test_schema_parses_as_draft202012(schema_path: Path) -> None:
    """Verify each schema file is a valid Draft 2020-12 JSON Schema.

    Checks:
    1. File loads as valid UTF-8 JSON
    2. Contains $schema field matching Draft 2020-12 URI
    3. Resolves to Draft202012Validator
    4. Passes schema self-validation via check_schema()
    """
    # Load and parse JSON
    try:
        content = schema_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        pytest.fail(f"{schema_path.name}: Failed to read as UTF-8: {e}")

    try:
        schema = json.loads(content)
    except json.JSONDecodeError as e:
        pytest.fail(f"{schema_path.name}: Invalid JSON: {e}")

    # Verify $schema field exists and matches Draft 2020-12
    if "$schema" not in schema:
        pytest.fail(f"{schema_path.name}: Missing '$schema' field")

    if schema["$schema"] != EXPECTED_SCHEMA_URI:
        pytest.fail(
            f"{schema_path.name}: Expected $schema='{EXPECTED_SCHEMA_URI}', "
            f"got '{schema['$schema']}'"
        )

    # Verify validator_for resolves to Draft202012Validator
    validator_cls = validator_for(schema)
    if validator_cls is not Draft202012Validator:
        pytest.fail(
            f"{schema_path.name}: Expected Draft202012Validator, got {validator_cls.__name__}"
        )

    # Validate schema itself is well-formed
    try:
        validator_cls.check_schema(schema)
    except (SchemaError, Exception) as e:
        pytest.fail(f"{schema_path.name}: Schema self-validation failed: {e}")


def test_at_least_one_schema_exists() -> None:
    """Ensure specs/ directory contains at least one schema file."""
    schema_files = discover_schema_files()
    assert len(schema_files) > 0, (
        f"No *.schema.json files found in {SPECS_DIR}. Expected at least one schema file."
    )
