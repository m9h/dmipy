#!/bin/bash
.venv/bin/pytest dmipy_jax/tests/test_invariants.py dmipy_jax/tests/test_tt.py > test_output.log 2>&1
echo "Done" >> test_output.log
