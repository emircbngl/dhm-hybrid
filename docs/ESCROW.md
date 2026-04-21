**DRAFT v1.0 — 2026-04-20** — Not executed. Subject to final written agreement. This document is for negotiation discussion.

# Source Code Escrow — DHM Reconstruction v1.0

## 1. Purpose

This document sets out the terms under which Hybrid Optics ("Vendor") will deposit source and build materials for the DHM Reconstruction product with an independent escrow agent. The purpose is to hedge Customer's ongoing scientific operations: if Vendor ceases to exist as a going concern, is acquired without a successor committed to continuing support, or materially fails to meet its support obligations, Customer can obtain the materials needed to continue **using, maintaining, and internally modifying** the Product without dependence on Vendor.

Cross-references: this escrow is funded and described commercially in [COMMERCIAL.md](./COMMERCIAL.md) Sections 3 and 12. Release triggers interact with support obligations defined in [SLA.md](./SLA.md).

## 2. Deposit Materials

Each deposit (initial and each update) shall contain, at minimum:

| Item | Contents |
|---|---|
| Full source tree | `src/`, `tools/`, `tests/`, configuration files, and any build glue under version control |
| Build scripts | Scripts and instructions required to produce a signed, notarized `.app` bundle from source on macOS Apple Silicon |
| Dependency lockfile | Exact pinned versions of Python, PySide6, NumPy, SciPy, scikit-image, and all transitive dependencies sufficient for a reproducible build |
| Signing keys | macOS Developer ID certificate(s) and notarization credentials for the then-current release (held under additional access controls — see Section 8) |
| Deployment instructions | Step-by-step install, configuration, and IQ-style validation instructions for a new macOS Apple Silicon workstation |
| Contact info | Names and current contact details for Vendor engineering staff with knowledge of the codebase, at the time of each deposit |

## 3. Escrow Agent

The escrow agent shall be selected from the following shortlist, or a mutually agreed alternative:

- Iron Mountain Intellectual Property Management
- NCC Group Escrow
- EscrowTech International

Vendor pays the escrow agent's setup and annual fees (confirmed in [COMMERCIAL.md](./COMMERCIAL.md) Section 3). Customer pays any fees arising from Customer-specific requests (e.g. additional verification rounds beyond the annual baseline).

## 4. Release Conditions

The agent shall release the deposit to Customer upon written certification that **any one** of the following has occurred:

1. **Vendor insolvency** — Vendor files for bankruptcy, enters administration, is placed into liquidation, or ceases trading as a going concern.
2. **Acquisition without successor** — Vendor is acquired and the acquirer fails to affirm in writing, within 60 days of close, its intention to continue supporting the Product under substantially equivalent terms to this agreement.
3. **Support failure** — Vendor fails to respond to a paid, validly submitted support ticket for a period exceeding **90 calendar days** `[NEGOTIABLE]`, notwithstanding Customer having paid the current Annual Support & Maintenance fee in full.

## 5. Release Mechanics

1. Customer submits a written release request to the agent, citing the applicable trigger in Section 4 with supporting evidence.
2. Agent notifies Vendor (or, in the case of insolvency, the relevant administrator) in writing.
3. Vendor has **30 calendar days** to cure the alleged condition or formally contest the request.
4. If Vendor fails to cure or the cure is inadequate, the agent releases the deposit materials to Customer.
5. Upon release, Customer receives a **perpetual, worldwide, royalty-free license to use, compile, modify, and maintain** the released materials **solely for Customer's internal scientific operations**. Redistribution, sublicensing, or commercial resale of the released source remains prohibited.

## 6. Deposit Cadence

- **Initial deposit:** within 30 days of Customer's first commercial installation under [COMMERCIAL.md](./COMMERCIAL.md).
- **Update deposits:** on every minor release in the 1.x series (e.g. 1.1, 1.2) and on any release designated "major" by Vendor. Patch releases (1.0.1, 1.0.2, …) are not required to trigger a re-deposit unless they change build scripts or dependency pinning.
- **Maximum staleness:** no deposit shall be more than **12 months** old at any time, regardless of release cadence.

## 7. Verification

Once per calendar year, the agent (or a mutually agreed independent third party) shall perform a **compile check** of the most recent deposit. The check confirms:

- Source tree is complete and readable.
- Build scripts execute to completion on a clean macOS Apple Silicon reference environment.
- Resulting `.app` bundle launches and reports a version string matching the deposit manifest.

The verification does **not** include functional or scientific testing of the Product. A one-page verification report is delivered to both Vendor and Customer. Vendor pays for the annual verification as part of agent fees.

## 8. Confidentiality and Access Controls

- All deposit materials are held by the agent under a standard escrow NDA; the agent does not disclose contents to Customer absent a valid release.
- Signing keys are held in a separate sealed envelope or equivalent logical partition, accessible only on release.
- Customer personnel who receive released materials are bound by written confidentiality obligations at least as protective as those in [COMMERCIAL.md](./COMMERCIAL.md).
- Released materials may not be shared outside the Customer's internal engineering and scientific staff with a legitimate need to know.
