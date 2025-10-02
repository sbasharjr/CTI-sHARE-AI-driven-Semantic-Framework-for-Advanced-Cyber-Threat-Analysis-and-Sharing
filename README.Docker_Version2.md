# Docker & Semantic Framework for Advanced Threat Analysis and Sharing

This project integrates a semantic framework (such as [OpenCTI](https://www.opencti.io/), [STIX](https://oasis-open.github.io/cti-documentation/), and [TAXII](https://oasis-open.github.io/cti-documentation/taxii/)) for advanced threat analysis and sharing. The setup is containerized using Docker and Docker Compose.

## How It Works

- **`Dockerfile`**: Builds the main threat analysis service, installing the required semantic framework libraries (see `requirements.txt`).
- **`compose.yaml`**: Orchestrates the semantic analysis service and OpenCTI platform for sharing threat intelligence.
- **`.dockerignore`**: Keeps your Docker images lean by excluding unnecessary files.
- **`requirements.txt`**: Add required Python libraries, such as `stix2`, `taxii2-client`, `opencti-client`, etc.

## Usage

1. **Build and Run**
   ```bash
   docker compose up --build
   ```
2. **Access Services**
   - Threat analysis API (local): `http://localhost:5000`
   - OpenCTI platform: `http://localhost:8080`

3. **Configure Environment Variables**
   - Change `OPENCTI_API_TOKEN` and other secrets in `compose.yaml`.

## Extending

- Add your threat analysis logic in `main.py`.
- Use semantic framework libraries to analyze and share threats.

## Reference

- [OpenCTI Documentation](https://www.opencti.io/docs/)
- [STIX & TAXII Documentation](https://oasis-open.github.io/cti-documentation/)