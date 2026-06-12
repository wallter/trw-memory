# Security Policy

We take the security of trw-memory seriously. This document explains which
versions receive security fixes and how to report a vulnerability privately.

## Supported Versions

Security fixes are provided for the current minor release line. Older minor
versions are not maintained — please upgrade to the latest release before
reporting.

| Version | Supported          |
| ------- | ------------------ |
| 0.9.x   | :white_check_mark: |
| < 0.9   | :x:                |

## Reporting a Vulnerability

Please do **not** open a public GitHub issue for security vulnerabilities.

Instead, report privately by email to **security@trwframework.com**. Include
as much detail as you can:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, or a proof-of-concept, if available.
- The affected version(s) and your environment.
- Any suggested remediation.

We will acknowledge your report within **3 business days** and keep you
informed as we investigate and work toward a fix. We may ask for additional
information to reproduce or validate the issue.

## Scope

This policy covers the `trw-memory` package — its source code, the memory
engine, storage backends, and security modules it ships. Vulnerabilities in
third-party dependencies should be reported to their respective maintainers,
though we appreciate a heads-up so we can update our dependency pins.

## Safe Harbor

We support responsible, good-faith security research. If you make a good-faith
effort to comply with this policy during your research, we will consider your
research authorized, we will work with you to understand and resolve the issue
quickly, and we will not pursue or support legal action against you. Please
avoid privacy violations, data destruction, and service disruption, and only
interact with accounts you own or have explicit permission to access.

Thank you for helping keep trw-memory and its users safe.
