"""Key derivation and namespace isolation tests for trw_memory.security.encryption."""

from __future__ import annotations

import hashlib

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from trw_memory.security import (
    decrypt_field,
    derive_namespace_key,
    derive_namespace_key_bytes,
    encrypt_field,
    generate_master_key,
)

from ._test_encryption_support import _KEY_LENGTH


class TestGenerateMasterKey:
    def test_returns_32_bytes(self) -> None:
        key = generate_master_key()
        assert isinstance(key, bytes)
        assert len(key) == _KEY_LENGTH

    def test_each_call_is_unique(self) -> None:
        keys = {generate_master_key() for _ in range(20)}
        assert len(keys) == 20

    def test_returns_bytes_type(self) -> None:
        key = generate_master_key()
        assert type(key) is bytes

    def test_key_is_not_all_zeros(self) -> None:
        key = generate_master_key()
        assert key != b"\x00" * _KEY_LENGTH


class TestDeriveNamespaceKey:
    def test_deterministic_for_same_inputs(self) -> None:
        master = generate_master_key()
        k1 = derive_namespace_key(master, "agents")
        k2 = derive_namespace_key(master, "agents")
        assert k1 == k2

    def test_different_namespaces_produce_different_keys(self) -> None:
        master = generate_master_key()
        k_agents = derive_namespace_key(master, "agents")
        k_projects = derive_namespace_key(master, "projects")
        assert k_agents != k_projects

    def test_different_master_keys_produce_different_derived_keys(self) -> None:
        master1 = generate_master_key()
        master2 = generate_master_key()
        k1 = derive_namespace_key(master1, "namespace")
        k2 = derive_namespace_key(master2, "namespace")
        assert k1 != k2

    def test_derived_key_is_64_char_hex(self) -> None:
        master = generate_master_key()
        key = derive_namespace_key(master, "test-ns")
        assert len(key) == 64
        assert key == key.lower()
        assert set(key) <= set("0123456789abcdef")

    def test_rejects_short_master_key(self) -> None:
        with pytest.raises(ValueError, match="master_key must be 32 bytes"):
            derive_namespace_key(b"too_short", "namespace")

    def test_rejects_long_master_key(self) -> None:
        with pytest.raises(ValueError, match="master_key must be 32 bytes"):
            derive_namespace_key(b"x" * 64, "namespace")

    def test_empty_namespace_works(self) -> None:
        master = generate_master_key()
        key = derive_namespace_key(master, "")
        assert len(key) == 64

    def test_unicode_namespace_works(self) -> None:
        master = generate_master_key()
        key = derive_namespace_key(master, "namespace-\u00e9toile")
        assert len(key) == 64

    @pytest.mark.parametrize(
        "namespace",
        ["agents", "teams", "orgs", "global", "project-xyz", "a" * 200],
    )
    def test_parametrized_namespace_variants(self, namespace: str) -> None:
        master = generate_master_key()
        key = derive_namespace_key(master, namespace)
        assert isinstance(key, str)
        assert len(key) == 64

    def test_matches_prd_hkdf_parameters(self) -> None:
        master = bytes(range(32))
        namespace = "global"
        key = derive_namespace_key(master, namespace)
        reference = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=hashlib.sha256(namespace.encode("utf-8")).digest(),
            info=b"trw-memory-namespace-key-v1",
        ).derive(master)

        assert key == reference.hex()
        assert derive_namespace_key_bytes(master, namespace) == reference


class TestHkdfNamespaceIsolation:
    def test_same_master_different_namespaces_cannot_cross_decrypt(self) -> None:
        master = generate_master_key()
        key_a = derive_namespace_key_bytes(master, "namespace-A")
        key_b = derive_namespace_key_bytes(master, "namespace-B")

        ciphertext = encrypt_field("secret for A only", key_a)
        with pytest.raises(InvalidTag):
            decrypt_field(ciphertext, key_b)

    def test_derived_keys_are_distinct_for_all_standard_namespaces(self) -> None:
        master = generate_master_key()
        namespaces = ["default", "agents", "teams", "orgs", "global"]
        keys = [derive_namespace_key(master, ns) for ns in namespaces]
        assert len(set(keys)) == len(namespaces)

    def test_long_namespace_string_derives_valid_key(self) -> None:
        master = generate_master_key()
        long_ns = "a" * 1000
        key = derive_namespace_key_bytes(master, long_ns)
        assert len(key) == _KEY_LENGTH
        ct = encrypt_field("data", key)
        assert decrypt_field(ct, key) == "data"

    def test_namespace_key_derivation_is_independent_of_content(self) -> None:
        master = generate_master_key()
        key = derive_namespace_key_bytes(master, "test-ns")
        ct1 = encrypt_field("content A", key)
        ct2 = encrypt_field("content B", key)
        assert ct1 != ct2
        assert decrypt_field(ct1, key) == "content A"
        assert decrypt_field(ct2, key) == "content B"
