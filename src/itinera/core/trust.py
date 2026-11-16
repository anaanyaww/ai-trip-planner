"""TLS trust configuration.

Python's HTTP libraries verify certificates against ``certifi``, a bundle of public
certificate authorities that ships with the package. That is the right default on a
plain network and the wrong one behind TLS inspection: corporate proxies, VPN clients
and several antivirus products terminate TLS and re-sign it with a private root. That
root is installed in the operating system's trust store, where the OS and browsers find
it, and is absent from ``certifi``, where Python looks. Every outbound call then fails
with ``CERTIFICATE_VERIFY_FAILED`` - search, weather, currency and every model provider
alike - on a machine where the same URLs open fine in a browser.

Delegating verification to the operating system fixes all of them at once, and costs
nothing on a machine with no interception, where the OS store contains the same public
roots ``certifi`` does. This is a change in *where the trusted roots come from*, not a
weakening of verification: certificates are still validated, and an untrusted one is
still rejected.

The patch is applied globally rather than passed to each client because the provider
SDKs build their own HTTP clients internally and do not all expose a way to inject one.
"""

from __future__ import annotations

import truststore

from itinera.core.config import Settings
from itinera.core.logging import get_logger

logger = get_logger(__name__)

_applied = False


def enable_system_trust_store(settings: Settings) -> bool:
    """Verify TLS against the operating system's trust store.

    Must run before any HTTP client is constructed, since clients capture an SSL
    context when they are built.

    Args:
        settings: Supplies the enable flag, so a deployment that needs the bundled
            certificate authorities can turn this off.

    Returns:
        Whether the OS trust store is now in use.
    """
    global _applied

    if not settings.ssl_use_system_trust_store:
        logger.info("tls.system_trust_store_disabled")
        return False

    if _applied:
        return True

    truststore.inject_into_ssl()
    _applied = True
    logger.info("tls.system_trust_store_enabled")
    return True
