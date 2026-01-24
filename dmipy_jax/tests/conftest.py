from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser
import pytest
import jax

jax.config.update("jax_enable_x64", True)

# Initialize Sybil with PythonCodeBlockParser
pytest_collect_file = Sybil(
    parsers=[
        PythonCodeBlockParser(),
    ],
    patterns=["*.md"],
).pytest()
