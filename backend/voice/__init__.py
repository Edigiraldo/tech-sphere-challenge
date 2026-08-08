"""Voice adapters — STT and TTS provider abstractions.

This package exposes the common interfaces (Protocols, configuration, and
result types) shared across voice I/O adapters so that callers such as
``backend/conversation/`` never depend on a specific STT or TTS provider.
"""
