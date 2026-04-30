# AI Development Constitution

## General Guidelines
1. **Accuracy & Quality**: Ensure all code changes are functional, secure, and follow project-specific conventions.
2. **Context Awareness**: Before implementing changes, review existing code and the `changelog/` directory to maintain consistency.
3. **Atomic Changes**: Commit or document changes in small, logical increments.

## Changelog Protocol (Speckit-style)
You are responsible for tracking and documenting every modification made to the codebase within the `changelog/` directory.

### Management Rules
- **Classification**: Categorize every change into one of the following files:
    - `changelog/feat.md`: New features or significant functional additions.
    - `changelog/bug.md`: Bug fixes and error handling improvements.
    - `changelog/perf.md`: Performance optimizations and refactors for efficiency.
- **Template Adherence**: All new entries must strictly follow the format defined in `changelog/.template.md`.
- **Automated Updates**: Immediately after a task is completed, append the change details to the appropriate file. Ensure the entry includes a concise description of the change and its impact.

## Operation
When prompted to implement a feature or fix, your workflow must be:
1. Implement the requested code change.
2. Identify the change type (Feature, Bug, or Performance).
3. Update the corresponding markdown file in `changelog/` following the `.template.md` structure.