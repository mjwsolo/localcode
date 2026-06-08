# Security Policy

## Reporting

If you find a security issue in LocalCode, please do not open a public GitHub issue first.

Report it privately via [GitHub Security Advisories](https://github.com/mjwsolo/localcode/security/advisories/new).

Include:

- affected version or commit
- reproduction steps
- potential impact
- any proposed mitigation

## Scope

Security-sensitive areas include:

- shell execution
- file permissions and approvals
- background jobs and daemons
- browser or MCP integrations
- local data storage and session history
- secrets/config handling

## Response Goals

Best effort:

- acknowledge receipt within 7 days
- confirm severity after reproduction
- publish a fix or mitigation as quickly as practical

## User Guidance

Until the project stabilizes further:

- review commands before approving them
- avoid running LocalCode against sensitive repos without understanding the configured permissions
- keep model/provider credentials in environment variables or local config, not committed files
