**DRAFT v1.0 — 2026-04-20** — Not executed. Subject to final written agreement. This document is for negotiation discussion.

# Service Level Agreement — DHM Reconstruction v1.0

## 1. Scope

This SLA applies to the DHM Reconstruction Product in production use at Customer's Site during an active Support Term, as defined in [COMMERCIAL.md](./COMMERCIAL.md) Sections 2 and 6. It covers Vendor's response and resolution commitments for defects, degradations, and operational questions directly attributable to the Product. Escrow-related obligations are handled separately under [ESCROW.md](./ESCROW.md).

## 2. Severity Matrix

Severity is assigned by Vendor on first triage, with Customer's input. Escalation or de-escalation is negotiated in good faith.

| Severity | Definition | Acknowledgement | Workaround | Fix |
|---|---|---|---|---|
| **S1** | Production down, data loss risk, or Product unusable across the Site | **4 business hours** `[NEGOTIABLE]` | 1 business day | 10 business days |
| **S2** | Major feature impaired; no acceptable workaround available | Next business day | 5 business days | 30 business days |
| **S3** | Minor feature impaired; an acceptable workaround exists | 3 business days | — | Next minor release |
| **S4** | Cosmetic, documentation, or enhancement request | Best effort | — | Backlog / discretionary |

Definitions:
- **Acknowledgement** — Vendor confirms receipt, assigns a ticket number, and names a responsible engineer.
- **Workaround** — any Vendor-advised procedure that restores operation, even if suboptimal.
- **Fix** — a code change delivered through the signed Update Channel described in [COMMERCIAL.md](./COMMERCIAL.md) Section 9.

## 3. Business Hours

- **Vendor business hours:** 09:00–18:00 **Europe/Istanbul**, Monday–Friday, excluding official Turkish national holidays (list published annually).
- **Customer timezone alignment:** **TBD** `[NEGOTIABLE]` — to be confirmed against Customer's primary operational timezone during contracting. For customers outside ±3h of Istanbul, the parties will agree in writing on either (a) an extended coverage window, or (b) an explicit expectation that S1 response clocks against Istanbul business hours.

All commitments in the Severity Matrix are measured in **business hours / business days** as defined above unless otherwise noted.

## 4. Support Channels

| Channel | Use |
|---|---|
| Email | All severities. Address: **`support@[vendor-domain].tbd`** |
| Web ticketing | All severities. Portal: **`https://[vendor-portal].tbd`** (placeholder) |
| Phone escalation | **S1 only.** Available 24×5 within the customer-timezone business window. Number provided on contract signature. |

Tickets must include: Product version, macOS version, affected Seat hardware UUID, reproduction steps, relevant log files from `~/Library/Logs/DHM-Reconstruction/`, and a designated Customer point of contact.

## 5. Exclusions

This SLA does **not** cover:

- Custom feature development, integrations, or bespoke analysis work not in the shipped Product.
- Training beyond the initial 2 × 4h remote session included in [COMMERCIAL.md](./COMMERCIAL.md).
- Defects caused by macOS operating system upgrades, Apple hardware faults, or changes to third-party camera / microscope drivers.
- Issues arising from Customer modification of installation binaries, configuration, or runtime environment outside documented supported paths.
- Acts of God, force majeure events, cyber attacks on Customer infrastructure, or sustained network outages affecting remote diagnostics.
- End-of-life or security-driven removal of third-party dependencies upstream. Vendor will make reasonable effort to adapt but is not bound by standard fix timelines in such cases.

## 6. Reporting

Vendor delivers a **quarterly support-performance report** to Customer within 15 business days of each calendar quarter-end. The report covers:

- Ticket volume by severity.
- Acknowledgement and resolution times versus SLA commitments.
- Any missed commitments and root-cause notes.
- Planned patch / minor release calendar for the coming quarter.

## 7. Service Credits

If Vendor misses the **S1 acknowledgement commitment** (Section 2) on **two or more tickets within a rolling calendar quarter**, Customer is entitled to a **10% credit** `[NEGOTIABLE]` applied against the next Annual Support & Maintenance invoice.

- Credit is computed on the Annual Support fee only, not on license fees.
- Maximum credit in any single support year: **20%** `[NEGOTIABLE]` of that year's Annual Support fee.
- Credits are the Customer's **sole and exclusive financial remedy** for SLA misses, without prejudice to Customer's termination rights in [COMMERCIAL.md](./COMMERCIAL.md) Section 10.
- Credit claims must be submitted in writing within 30 days of the end of the quarter in which the breach occurred.
