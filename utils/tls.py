"""Use the operating system's certificate store for HTTPS.

Antivirus HTTPS scanning (e.g. AVG/Avast Web Shield) and corporate proxies re-sign traffic with
a root certificate that Windows trusts but Python's bundled list does not. truststore makes
Python verify against the OS store instead, so verification stays on and still works.
"""


def use_system_certificates() -> None:
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()
