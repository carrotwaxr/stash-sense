"""Face recognition failure analysis (issue #139).

Runs the production scene-identification pipeline over a fixed benchmark set
drawn from the user's own StashDB-linked scenes, and gives every missed
performer exactly one reason. See docs/research/failure-analysis.md.

Entry point: ``python -m benchmark.failure --help``.
"""
