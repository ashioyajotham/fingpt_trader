from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ValidationError:
    field: str
    message: str


class DataValidator:
    def __init__(self, schema: Dict):
        self.schema = schema

    def validate(self, data: Dict) -> List[ValidationError]:
        """Validate data against schema"""
        errors = []

        for field, rules in self.schema.items():
            value = data.get(field)
            field_errors = self._validate_field(field, value, rules)
            errors.extend(field_errors)

        return errors

    def _validate_field(
        self, field: str, value: Any, rules: Dict
    ) -> List[ValidationError]:
        """Validate single field"""
        errors = []

        if rules.get("required", False) and value is None:
            errors.append(ValidationError(field, "Field is required"))

        if value is not None:
            if "type" in rules and not isinstance(value, rules["type"]):
                errors.append(ValidationError(field, f"Expected type {rules['type']}"))

            if "range" in rules:
                min_val, max_val = rules["range"]
                if value < min_val or value > max_val:
                    errors.append(
                        ValidationError(
                            field, f"Value out of range [{min_val}, {max_val}]"
                        )
                    )

        return errors
