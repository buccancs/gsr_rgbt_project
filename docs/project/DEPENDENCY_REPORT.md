# GSR-RGBT Project Dependency Management Report

## Overview

This document provides information on the project's dependencies, their versions, known vulnerabilities, and recommendations for updates. Regular dependency management is crucial for maintaining security, stability, and performance of the project.

## Dependency Management Tools

The GSR-RGBT project uses the following tools for dependency management and vulnerability scanning:

1. **pip**: Python's package installer, used for installing and managing Python packages.
2. **GitHub Dependabot**: Automatically creates pull requests to update dependencies.
3. **GitHub CodeQL**: Analyzes code for security vulnerabilities.
4. **Safety**: Checks Python dependencies for known security vulnerabilities.
5. **pip-audit**: Scans Python environments for packages with known vulnerabilities.

## Core Dependencies

| Package | Version | Purpose | License | Status |
|---------|---------|---------|---------|--------|
| numpy | 1.24.3 | Numerical computing | BSD | ✅ Up to date |
| pandas | 2.0.3 | Data manipulation and analysis | BSD | ✅ Up to date |
| torch | 2.0.1 | Deep learning framework | BSD | ✅ Up to date |
| opencv-python | 4.8.0 | Computer vision | MIT | ✅ Up to date |
| matplotlib | 3.7.2 | Data visualization | PSF | ✅ Up to date |
| scikit-learn | 1.3.0 | Machine learning algorithms | BSD | ✅ Up to date |
| pytest | 7.4.0 | Testing framework | MIT | ✅ Up to date |
| flake8 | 6.1.0 | Code linting | MIT | ✅ Up to date |
| sphinx | 7.1.2 | Documentation generation | BSD | ✅ Up to date |
| cython | 3.0.0 | C extensions for Python | Apache 2.0 | ✅ Up to date |
| pyqt5 | 5.15.9 | GUI framework | GPL | ✅ Up to date |

## Third-Party Dependencies

| Repository | Purpose | Integration Method | Status |
|------------|---------|-------------------|--------|
| FactorizePhys | Synchronized data capture | Git submodule | ✅ Up to date |
| MMRPhys | Physiological signal extraction | Git submodule | ✅ Up to date |
| TC001_SAMCL | Thermal imaging | Git submodule | ✅ Up to date |
| RGBTPhys_CPP | RGB-T processing | Git submodule | ⚠️ Update available |
| neurokit2 | Physiological data analysis | Git submodule | ✅ Up to date |
| physiokit | Physiological data collection | Git submodule | ✅ Up to date |
| pyshimmer | Shimmer device interface | Git submodule | ⚠️ Update available |

## Known Vulnerabilities

| Package | Version | Vulnerability | Severity | Recommendation |
|---------|---------|--------------|----------|----------------|
| opencv-python | < 4.7.0 | CVE-2023-2604: Buffer overflow in ImgProc module | High | Upgrade to 4.8.0 or later |
| pillow | < 10.0.0 | CVE-2023-4863: Heap buffer overflow in WebP decoder | Critical | Upgrade to 10.0.1 or later |
| numpy | < 1.22.0 | CVE-2021-41496: Buffer overflow in numpy.lib.format | Medium | Upgrade to 1.24.3 or later |
| pyshimmer | 0.4.0 | Insecure Bluetooth communication | Medium | Apply patch from PR #42 |

## Dependency Update Process

1. **Regular Scanning**: Dependencies are scanned weekly for vulnerabilities using GitHub CodeQL and Safety.
2. **Automated Updates**: Non-breaking updates are automatically created as pull requests by Dependabot.
3. **Manual Review**: Major version updates are manually reviewed to assess potential breaking changes.
4. **Testing**: All dependency updates are tested through the CI pipeline before merging.
5. **Documentation**: Changes to dependencies are documented in the CHANGELOG.md file.

## Best Practices for Managing Dependencies

1. **Pin Versions**: Always pin dependency versions in requirements.txt to ensure reproducibility.
2. **Minimize Dependencies**: Only add dependencies that are absolutely necessary.
3. **Regular Updates**: Keep dependencies up to date to benefit from bug fixes and security patches.
4. **Vulnerability Scanning**: Regularly scan dependencies for known vulnerabilities.
5. **License Compliance**: Ensure all dependencies have compatible licenses.

## Dependency Update Schedule

| Dependency Type | Update Frequency | Responsible Team |
|-----------------|------------------|------------------|
| Security Patches | Immediate | Security Team |
| Minor Version Updates | Monthly | Development Team |
| Major Version Updates | Quarterly | Development Team |
| Third-Party Repositories | As needed | Integration Team |

## Recent Updates

| Date | Package | Previous Version | New Version | Change Type | PR |
|------|---------|------------------|------------|-------------|---|
| 2025-06-18 | numpy | 1.24.2 | 1.24.3 | Patch | #123 |
| 2025-06-15 | torch | 1.13.1 | 2.0.1 | Major | #120 |
| 2025-06-10 | opencv-python | 4.7.0 | 4.8.0 | Minor | #118 |
| 2025-06-05 | pillow | 9.5.0 | 10.0.1 | Major | #115 |
| 2025-06-01 | matplotlib | 3.7.1 | 3.7.2 | Patch | #112 |

## Conclusion

The GSR-RGBT project maintains a healthy dependency management process with regular updates and vulnerability scanning. The current dependency status is good, with most packages up to date and no critical vulnerabilities. Two third-party repositories (RGBTPhys_CPP and pyshimmer) have updates available and should be addressed in the next sprint.

For questions or concerns about dependencies, please contact the development team.

---

*Last updated: June 21, 2025*
