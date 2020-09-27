# Copyright (c) 2016 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import json
import unittest

import jsonschema
from optlang.glpk_interface import Variable, Constraint, Objective, Model

bound_schema = {
    "oneOf": [
        {"type": "number"},
        {"type": "null"}
    ]
}

variable_schema = {
    "$schema": "variable",
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
        },
        "lb": bound_schema,
        "ub": bound_schema,
        "type": {
            "type": "string",
            "enum": ["continuous", "binary", "integer"]
        }
    },
    "required": ["name", "lb", "ub", "type"]
}

expression_definition = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
        },
        "type": {
            "type": "string",
            "enum": ["Add", "Mul", "Pow", "Number", "Symbol"]
        },
        "value": {
            "type": "number"
        },
        "args": {
            "type": "array",
            "items": {"$ref": "#/definitions/expr"}
        }
    },
    "required": ["type"]
}

constraint_schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
        },
        "lb": bound_schema,
        "ub": bound_schema,
        "indicator_variable": {
            "oneOf": [
                {"type": "string"},
                {"type": "null"}
            ]
        },
        "active_when": {
            "oneOf": [
                {"type": "number",
                 "enum": [0, 1]},
                {"type": "null"}
            ]
        },
        "expression": {"$ref": "#/definitions/expr"}
    },
    "definitions": {
        "expr": expression_definition
    }
}

objective_schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
        },
        "direction": {
            "type": "string",
            "enum": ["min", "max"]
        },
        "expression": {"$ref": "#/definitions/expr"}
    },
    "definitions": {
        "expr": expression_definition
    }
}

model_schema = {
    "type": "object",
    "properties": {
        "name": {
            "oneOf": [
                {"type": "string"},
                {"type": "null"}
            ]
        },
        "objective": objective_schema,
        "variables": {
            "type": "array",
            "items": variable_schema
        },
        "constraints": {
            "type": "array",
            "items": constraint_schema
        }
    },
    "definitions": {
        "expr": expression_definition
    }
}


class JsonTest(unittest.TestCase):
    def setUp(self):
        self.var1 = var1 = Variable("var1", lb=0, ub=1, type="continuous")
        self.var2 = var2 = Variable("var2", lb=0, ub=1, type="continuous")
        self.const1 = const1 = Constraint(0.5 * var1, lb=0, ub=1, name="c1")
        self.const2 = const2 = Constraint(0.1 * var2 + 0.4 * var1, name="c2")
        self.model = model = Model()
        model.add([var1, var2])
        model.add([const1, const2])
        model.objective = Objective(var1 + var2)
        model.update()
        self.json_string = json.dumps(model.to_json())

    def test_model_json_validates(self):
        jsonschema.validate(self.model.to_json(), model_schema)

    def test_objective_json_validates(self):
        jsonschema.validate(self.model.objective.to_json(), objective_schema)

    def test_model_is_reconstructed_from_json(self):
        model = Model.from_json(json.loads(self.json_string))
        self.assertEqual(model.variables["var1"].lb, 0)
        self.assertEqual(model.variables["var1"].ub, 1)
        self.assertEqual(model.constraints["c1"].expression, 0.5 * model.variables["var1"])
        self.assertEqual(
            (model.objective.expression - sum(model.variables)).simplify(), 0
        )


if __name__ == "__main__":
    import nose

    nose.runmodule()
