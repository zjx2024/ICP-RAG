"""Storage backends.

Submodules are intentionally not imported eagerly: importing the filesystem
artifact backend must not connect to PostgreSQL, OpenSearch, or HugeGraph.
Callers should import the concrete client from its defining module.
"""
