"""
SSL/TLS Context provides secure network communication.

Plain TCP sends bytes that anyone sitting on the network path can read (like a
postcard). TLS (Transport Layer Security, the modern name for SSL, and the S
in HTTPS) turns the postcard into a sealed envelope. Data is encrypted in
transit, and you can be sure you are really talking to the intended party, not
an impostor.

An ssl.SSLContext is a settings bundle describing how the encryption should
happen: which certificates to trust, which certificate to present, which TLS
versions and ciphers are acceptable. You create one context and reuse it for
many connections. Asyncio uses it during the handshake (the initial
back-and-forth where the two sides verify certificates and agree on encryption
keys before any real data flows).

Key terms:
- Certificate: A digital ID card ("I am example.com") signed by a trusted
  authority. Servers almost always present one. Clients verify it.
- CA (Certificate Authority): A trusted organization that signs certificates.
  ssl.create_default_context() loads your system's built-in CA list. So
  certificates from real websites verify automatically.
- Verify / check_hostname: The client's safety checks. Is this certificate
  signed by a CA I trust? And is it actually for the host I dialed? Only turn
  these off for local testing (e.g. with self-signed certs). Never in
  production.
- Self-signed certificate: An ID card you made yourself. Fine for local demos.
  Browsers/clients will not trust it by default.
- mTLS (mutual TLS): Both sides show ID. The server also demands a client
  certificate (require_client_cert here). Used in service-to-service auth.

When to use:
Whenever your TCP client/server crosses an untrusted network, or handles any
data you would not print on a postcard. Locally you can skip TLS. Over the
internet you almost always need it.

Minimal example:
    import asyncio
    from asyncio_mastery.streams.ssl_context import SSLContextManager

    async def main() -> None:
        ctx = SSLContextManager.create_client_context()
        reader, writer = await asyncio.open_connection(
            "example.com", 443, ssl=ctx, server_hostname="example.com"
        )
        writer.write(b"GET / HTTP/1.0\\r\\nHost: example.com\\r\\n\\r\\n")
        await writer.drain()
        print(await reader.read(200))
        writer.close()
        await writer.wait_closed()

    asyncio.run(main())
"""

import asyncio
import os
import ssl
import tempfile


class SSLContextManager:
    """Manages SSL contexts for secure connections.

    Provides factory methods for the two sides of a TLS connection:
    - Client context: Knows which certificates to trust (verification on).
    - Server context: Knows which certificate+private key to present.

    Also caches one context of each kind. Creating a context parses certificate
    files, which is relatively expensive. Reusing one context across all your
    connections is both faster and the recommended pattern.

    Example:
        mgr = SSLContextManager()
        client_ctx = mgr.get_client_context()
        server_ctx = mgr.get_server_context("cert.pem", "key.pem")
    """

    def __init__(self) -> None:
        """Create an empty manager. Contexts are built lazily on first use."""
        self._client_context: ssl.SSLContext | None = None
        self._server_context: ssl.SSLContext | None = None

    @staticmethod
    def create_client_context(
        *,
        verify: bool = True,
        ca_file: str | None = None,
        cert_file: str | None = None,
        key_file: str | None = None,
    ) -> ssl.SSLContext:
        """Create SSL context for client connections.

        Builds a context for the side that dials out. By default it trusts the
        system CA store and verifies both the certificate's signature chain and
        that the certificate matches the hostname you connected to. These two
        checks stop impersonators.

        Args:
            verify: Set False to skip certificate verification entirely. Only
                acceptable for local testing with self-signed certificates.
                Disabling it removes protection against man-in-the-middle
                attacks.
            ca_file: Optional path to a PEM file with extra/alternative CAs to
                trust (e.g. your company's internal CA).
            cert_file: Optional path to a client certificate, for servers that
                require mutual TLS (mTLS). Must be given together with key_file.
            key_file: Optional path to the private key matching cert_file.

        Returns:
            A ready-to-use ssl.SSLContext for asyncio.open_connection(...,
            ssl=context).

        Note:
            The context is pinned to TLS 1.2+ and modern AEAD ciphers
            (AES-GCM / ChaCha20-Poly1305 with ECDHE/DHE key exchange). Old,
            broken protocol versions (SSLv3, TLS 1.0/1.1) are refused.
        """
        context = ssl.create_default_context()

        if not verify:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        if ca_file:
            context.load_verify_locations(ca_file)

        if cert_file and key_file:
            context.load_cert_chain(cert_file, key_file)

        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20")

        return context

    @staticmethod
    def create_server_context(
        cert_file: str,
        key_file: str,
        *,
        ca_file: str | None = None,
        require_client_cert: bool = False,
    ) -> ssl.SSLContext:
        """Create SSL context for server.

        Builds a context for the side that accepts connections. The server must
        have a certificate + private key. That is the ID card it shows to every
        connecting client during the handshake.

        Args:
            cert_file: Path to the server's certificate (PEM format, i.e. the
                -----BEGIN CERTIFICATE----- text file).
            key_file: Path to the certificate's private key (PEM). Guard this
                file like a password. Anyone with it can impersonate you.
            ca_file: Optional CA bundle used to verify client certificates.
            require_client_cert: If True, clients must present a valid
                certificate (mutual TLS). Connections without one are rejected
                during the handshake. If False (with ca_file set), client
                certificates are requested but optional.

        Returns:
            A ready-to-use ssl.SSLContext for asyncio.start_server(...,
            ssl=context).
        """
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

        context.load_cert_chain(cert_file, key_file)

        if ca_file:
            context.load_verify_locations(ca_file)
            if require_client_cert:
                context.verify_mode = ssl.CERT_REQUIRED
            else:
                context.verify_mode = ssl.CERT_OPTIONAL

        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20")

        return context

    @staticmethod
    def generate_self_signed(
        *, common_name: str = "localhost", valid_days: int = 365, key_size: int = 2048
    ) -> tuple[str, str]:
        """Generate self-signed certificate for testing (returns cert_path, key_path).

        Creates a certificate that vouches for itself (no trusted authority
        signs it). Perfect for local demos and automated tests. But clients
        must use verify=False (or explicitly trust it) to connect.

        Requires the third-party cryptography package.

        Args:
            common_name: The hostname this certificate claims to represent
                (becomes the CN and a DNS Subject Alternative Name). Use
                "localhost" for local testing.
            valid_days: How many days until the certificate expires.
            key_size: RSA key size in bits. 2048 is the sane minimum.

        Returns:
            A (cert_file_path, key_file_path) tuple pointing at two new
            temporary PEM files. Delete them when done (they contain a private
            key).
        """
        import datetime

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)

        subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=valid_days))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(common_name)]), critical=False)
            .sign(key, hashes.SHA256())
        )

        cert_file = tempfile.mktemp(suffix=".pem")
        key_file = tempfile.mktemp(suffix=".key")

        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        with open(key_file, "wb") as f:
            f.write(
                key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        return cert_file, key_file

    def get_client_context(self) -> ssl.SSLContext:
        """Get or create default client context.

        Lazily builds one verifying client context with defaults and then
        reuses it for the lifetime of this manager. Context creation reads
        trust stores from disk, so caching avoids repeating that work.

        Returns:
            The shared default client ssl.SSLContext.
        """
        if self._client_context is None:
            self._client_context = self.create_client_context()
        return self._client_context

    def get_server_context(self, cert_file: str, key_file: str) -> ssl.SSLContext:
        """Get or create server context.

        Lazily builds one server context from the given certificate pair and
        then reuses it.

        Args:
            cert_file: Path to the server certificate (PEM).
            key_file: Path to the matching private key (PEM).

        Returns:
            The shared server ssl.SSLContext, built from those files.
        """
        if self._server_context is None:
            self._server_context = self.create_server_context(cert_file, key_file)
        return self._server_context


async def demonstrate_ssl() -> None:
    """Demonstrate SSL usage.

    Generates a throwaway self-signed certificate, starts an encrypted echo
    server, connects a client to it, exchanges one message through the
    encrypted tunnel, and cleans everything up. If the cryptography package
    isn't installed, it prints a usage summary instead. Run directly to try it.
    """
    print("=== SSL/TLS Demo ===")

    try:
        cert_file, key_file = SSLContextManager.generate_self_signed()
        print(f"Generated cert: {cert_file}")
        print(f"Generated key: {key_file}")

        server_ctx = SSLContextManager.create_server_context(cert_file, key_file)
        client_ctx = SSLContextManager.create_client_context(
            verify=False
        )

        print(f"Server context: {server_ctx}")
        print(f"Client context: {client_ctx}")

        async def ssl_server(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            data = await reader.read(100)
            writer.write(b"SSL Echo: " + data)
            await writer.drain()

        server = await asyncio.start_server(ssl_server, "127.0.0.1", 0, ssl=server_ctx)
        port = server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=client_ctx)

        writer.write(b"Hello SSL!")
        await writer.drain()
        response = await reader.read(100)
        print(f"SSL Response: {response.decode()}")

        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()

        os.unlink(cert_file)
        os.unlink(key_file)

    except ImportError:
        print("cryptography not installed, skipping cert generation")
        print("SSL Context methods:")
        print("  create_client_context() - for clients")
        print("  create_server_context() - for servers")
        print("  generate_self_signed() - for testing")

    print("\nSSL best practices:")
    print("  - Use TLS 1.2 minimum")
    print("  - Prefer ECDHE key exchange")
    print("  - Use AES-GCM or ChaCha20-Poly1305 ciphers")
    print("  - Verify hostnames in production")
    print("  - Rotate certificates regularly")


if __name__ == "__main__":
    asyncio.run(demonstrate_ssl())
