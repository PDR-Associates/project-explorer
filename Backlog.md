# Backlog

Future enhancements for the database surveyor and surrounding features. Phases 1–4
(core implementation, Egeria integration, web UI) are complete and merged.

## Database Surveyor — Phase 5+

- **Survey monitoring** — poll Egeria for native survey completion status instead of
  returning immediately on the async `survey_action_guid`.
- **Result comparison** — compare Egeria native annotations vs. the local custom scan
  side by side.
- **MySQL / Oracle support** — extend `surveyors/database/` beyond PostgreSQL
  (connection abstraction, introspection queries, Egeria templates).
- **Scheduled surveys** — automatic periodic re-survey of registered databases.
- **Data quality** — add quality checks to surveys (null rates, constraint coverage,
  anomaly detection).
- **Schema evolution** — track schema changes over time across survey runs.
- **Database comparison view** — side-by-side schema comparison in the web UI.
- **Schema visualization** — interactive ER diagrams from PK/FK metadata.

---

For the original design see `docs/database-surveyor-design.md`. Build-time phase
snapshots are kept in the (gitignored) `docs/archive/` folder.
