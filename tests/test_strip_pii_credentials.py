"""The egress redactor removes every credential shape, not only the prefix-token ones (PII-REDACTOR-GATING-CORPUS).

``strip_pii`` is what leaves the machine at the publish boundary (``sync/_remote_publish``) and on recall previews. It
masked emails and ``<prefix>-<token>`` keys but not a password assignment, a URL credential, a JWT, a PEM key block or a
hyphenated provider key, all of which ``mask_credentials`` already knew. Inputs are assembled at run time so this file
holds no key-shaped literal.
"""

from __future__ import annotations

import pytest

from trw_memory.security.credentials import mask_credentials
from trw_memory.security.pii import strip_pii

_ALPHA = "AbCdEfGhIjKlMnOpQrStUvWxYz"
_JWT = ".".join(
    (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0",
        "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    )
)
_HEX = "4f9c2a7e1b6d8305a9c4e2f7b1d60385"

_CASES = {
    "anthropic key": ("key sk-ant-api03-" + _ALPHA + "0123456789_-abcdefghij in the log", _ALPHA + "0123456789"),
    "google key": (
        "maps key AIza" + "SyA-abcdefghijklmnopqrstuvwxyz012345 in the config",
        "abcdefghijklmnopqrstuvwxyz012345",
    ),
    "password assignment": ("config password=hunter2hunter2 end", "hunter2hunter2"),
    "labelled hex": ("secret: " + _HEX + " in the env", _HEX),
    "url credentials": ("connect postgres://admin:s3cr3tpass@localhost:5432/app now", "s3cr3tpass"),
    "pem block": (
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAu1SU1LfVLPHCozMxH2Mo4lg\n-----END RSA PRIVATE KEY-----",
        "MIIEowIBAAKCAQEAu1SU1LfVLPHCozMxH2Mo4lg",
    ),
    "bearer jwt": ("Authorization: Bearer " + _JWT, _JWT),
    "cookie jwt": ("Cookie: theme=dark; session=" + _JWT, _JWT),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_strip_pii_removes_the_credential(name: str) -> None:
    text, secret = _CASES[name]
    assert secret not in strip_pii(text)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_mask_credentials_removes_the_credential(name: str) -> None:
    text, secret = _CASES[name]
    assert secret not in mask_credentials(text)


def test_strip_pii_keeps_ordinary_prose_and_is_idempotent() -> None:
    prose = "The token count is 42 and the secret to good tests is small steps; see abc123 in commit deadbeef."
    assert strip_pii(prose) == prose
    text, _ = _CASES["password assignment"]
    once = strip_pii(text)
    assert strip_pii(once) == once


@pytest.mark.parametrize("label", ["password=", "password: ", "secret: ", 'token: "'])
def test_a_hex_prefix_never_leaves_its_tail_in_the_clear(label: str) -> None:
    tail = "-tail-of-the-credential!"
    for redactor in (strip_pii, mask_credentials):
        out = redactor("x " + label + _HEX + tail + " y")
        assert _HEX not in out and "tail-of-the-credential" not in out


@pytest.mark.parametrize("quote", ['"', "'"])
def test_a_quoted_hex_value_is_masked_whole_even_with_spaces_inside(quote: str) -> None:
    for label in ("secret: ", "password= ", "token: "):
        text = "x " + label + quote + _HEX + " with trailing words" + quote + " y"
        for redactor in (strip_pii, mask_credentials):
            out = redactor(text)
            assert _HEX not in out and "trailing words" not in out


def test_a_labelled_hex_inside_a_quoted_env_value_does_not_split_the_quotes() -> None:
    from trw_memory.security.credentials import mask_low_confidence

    text = 'password="alpha privateword secret: ' + _HEX + '" end'
    for redactor in (strip_pii, mask_credentials, mask_low_confidence):
        out = redactor(text)
        assert _HEX not in out and "privateword" not in out and "alpha" not in out
