# Active Memory

- Date: 2026-02-07
- Status: Initialized project structure

## Phase 1: Setup & Mocking
- We are setting up the `aitigravity` microservice.
- The external databases (Auth, Order, Product) are NOT ready.
- **Goal:** Create a local PostgreSQL container that mocks the schemas defined in `docs/db-schema.md` so we can develop AI features without waiting for the main backend.
- **Next Step:** Implement `docker-compose.yml` with an initialization script to seed dummy data.