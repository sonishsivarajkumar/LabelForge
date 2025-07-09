# Security Policy

## Supported Versions

We actively support the following versions of LabelForge with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of LabelForge seriously. If you discover a security vulnerability, please follow these steps:

### Reporting Process

1. **Do not** report security vulnerabilities through public GitHub issues
2. Email security concerns to: security@labelforge.org (or create a private security advisory)
3. Include as much information as possible:
   - Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting)
   - Full paths of source file(s) related to the manifestation of the issue
   - The location of the affected source code (tag/branch/commit or direct URL)
   - Any special configuration required to reproduce the issue
   - Step-by-step instructions to reproduce the issue
   - Proof-of-concept or exploit code (if possible)
   - Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Acknowledgment**: We will acknowledge your email within 48 hours
- **Initial Response**: We will provide an initial response within 7 days
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days

### Disclosure Policy

- We ask that you give us a reasonable amount of time to address the issue before public disclosure
- We will coordinate with you on disclosure timing
- We will credit you in our security advisory (unless you prefer to remain anonymous)

## Security Measures

LabelForge implements several security measures:

### Code Security
- Regular dependency updates
- Static code analysis with Bandit
- Type checking with MyPy
- Automated security scanning in CI/CD

### Data Protection
- No collection of user data by default
- Local execution model
- Optional encrypted data storage
- Secure handling of sensitive labeling data

### Best Practices
- Regular security audits
- Secure coding guidelines for contributors
- Dependency vulnerability monitoring
- Security-focused code reviews

## Security-Related Configuration

When using LabelForge in production environments:

1. **Environment Isolation**: Use virtual environments or containers
2. **Access Control**: Implement appropriate access controls for data and models
3. **Network Security**: Use HTTPS for any web interfaces
4. **Data Encryption**: Encrypt sensitive data at rest and in transit
5. **Logging**: Monitor and log security-relevant events
6. **Updates**: Keep LabelForge and dependencies up to date

## Third-Party Dependencies

We regularly monitor our dependencies for known vulnerabilities using:
- GitHub Dependabot
- Automated security scanning
- Regular dependency updates

## Contact

For general security questions or concerns about LabelForge:
- Email: security@labelforge.org
- Create a private security advisory on GitHub

Thank you for helping keep LabelForge and our users safe!
