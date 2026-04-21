**DRAFT v1.0 — 2026-04-20** — Not executed. Subject to final written agreement. This document is for negotiation discussion.

# Commercial Terms — DHM Reconstruction v1.0

## 1. Purpose

These are draft commercial terms for the deployment of DHM Reconstruction v1.0 ("Product") by Hybrid Optics ("Vendor") to [Customer Name] ("Customer") for an initial Site of 5 Seats. This document reflects the Vendor's position as of the date above and is subject to final written agreement executed by authorized signatories of both parties. Cross-references: see [ESCROW.md](./ESCROW.md) for source escrow terms and [SLA.md](./SLA.md) for support response commitments.

## 2. Parties and Definitions

- **Vendor** — Hybrid Optics, developer and licensor of the Product.
- **Customer** — [Customer Name], the named licensee identified in the executed agreement.
- **Seat** — A single activated installation of the Product on one (1) Apple Silicon macOS workstation, identified by its hardware UUID ("Node-Locked").
- **Node-Locked License** — Non-transferable license bound to a specific hardware UUID; re-binding to replacement hardware is permitted once per Seat per 12 months at no charge.
- **Site** — A Customer-designated physical location (e.g. a single laboratory building) where Seats are deployed.
- **Support Term** — A rolling 12-month window during which the Customer has paid the Annual Support & Maintenance fee and is entitled to the services described in Section 6 and in [SLA.md](./SLA.md).
- **Update Channel** — The signed, notarized macOS `.app` bundle distribution mechanism described in Section 9; no over-the-network auto-update.

## 3. Pricing

| Item | Unit | Price (USD) | Notes |
|---|---|---|---|
| Perpetual License, Node-Locked | per Seat | **$17,500** `[NEGOTIABLE]` | CFO ceiling $16k; Vendor position $17.5k; comparable market: Nanolive $18k, Lyncée Tec $22k (with IQ/OQ) |
| Seat Minimum (new Customer, discount tier) | Seats | **5** `[NEGOTIABLE]` | Required to unlock this pricing tier |
| Annual Support & Maintenance | per Customer, flat | **$11,000 / year** `[NEGOTIABLE]` | Covers all Seats at the Site; not per-Seat |
| Source Escrow — Agent Fees | — | **Vendor** | See [ESCROW.md](./ESCROW.md) |
| Installation & Training (remote, 2 × 4h) | — | **Included** | For new Customers at initial deployment |

Total initial order (reference): 5 Seats × $17,500 = **$87,500** license + **$11,000** Year-1 Support = **$98,500** Year-1 total.

## 4. Payment Terms

Payments are structured as three milestone invoices, each payable Net 30 from invoice date:

| Milestone | Trigger | % of License Fee |
|---|---|---|
| M1 — Signature | Execution of agreement | **40%** `[NEGOTIABLE]` |
| M2 — IQ Acceptance | Successful IQ protocol, or 90 days after install (whichever is earlier) | **40%** `[NEGOTIABLE]` |
| M3 — Operational | 3 months of live operational use post-IQ | **20%** `[NEGOTIABLE]` |

The Annual Support & Maintenance fee is invoiced separately at agreement signature and annually thereafter on the anniversary.

Counter-position note: Customer CFO requested 30-60-90. Vendor's 40/40/20 front-loads delivery risk onto the Vendor (60% held until IQ + 3-month operational) while accelerating a portion of working capital. Negotiable on splits, not on the milestone structure.

## 5. License Grant

Vendor grants Customer a **perpetual, non-transferable, non-exclusive, Node-Locked License** to install and use the Product on the number of Seats purchased, at the designated Site, solely for Customer's internal scientific and research operations. The license is bound to a macOS Apple Silicon machine identified by its hardware UUID.

The license includes:
- All v1.0.x patch releases in perpetuity (e.g. 1.0.1, 1.0.2, …).
- Major and minor version upgrades (e.g. 1.1, 1.2, 2.0) **only during an active Support Term**.

Reverse engineering, redistribution, sublicensing, SaaS resale, and use in clinical diagnostic workflows are prohibited (see Section 11).

## 6. Support Term

The Support Term runs for 12 months from the invoice date of the Annual Support & Maintenance fee and is renewable.

**Included:**
- Bug fixes and security patches for the then-current major version.
- Remote diagnostics (screen share, log review).
- Acknowledgement of inbound support tickets within 1 business day (see [SLA.md](./SLA.md) for full severity matrix).
- Quarterly support-performance report.

**Excluded:**
- Hardware (Apple workstation, camera, microscope, storage).
- macOS operating system upgrades and Apple-side ecosystem changes.
- Custom feature development or integrations outside the standard Product.
- Training beyond the initial 2 × 4h session.

## 7. Warranty

Vendor warrants that for **90 days** `[NEGOTIABLE]` from the date of IQ acceptance, the Product will materially conform to its published specification for v1.0. Customer's exclusive remedy under this warranty is, at Vendor's option: (a) a bug fix delivered through the Update Channel, or (b) refund of a pro-rated portion of the affected Seat's license fee.

## 8. Indemnity

Vendor shall defend, indemnify, and hold Customer harmless from third-party claims alleging that the Product as delivered infringes a valid patent, copyright, or trade secret, provided Customer (a) gives prompt written notice, (b) grants Vendor sole control of the defense, and (c) cooperates reasonably.

**Aggregate cap:** the **lesser of fees paid by Customer to Vendor in the preceding 12 months, or USD $500,000** `[NEGOTIABLE]`.

> Counter-position: Customer Legal requested $2,000,000 IP indemnity cap. Vendor position is a dual cap (12-month fees OR $500k, whichever is lower). **Flag to PM: this is the single most likely Legal pushback item.** Middle ground likely lands at $1M or "fees paid in contract-to-date, capped at $1M."

## 9. Data Ownership and Telemetry

All data generated, processed, or stored by the Product — including holograms, reconstructions, phase maps, and analysis outputs — is the exclusive property of Customer and remains on Customer hardware and Customer-controlled network storage (NFS/SMB).

The Product:
- Transmits **no telemetry** to Vendor or any third party.
- Performs **no automatic update checks** over the network.
- Operates fully offline after installation.
- Reads central configuration from Customer-provided NFS/SMB paths only.

These properties are asserted in `SECURITY.md` (to be delivered with v1.0 release) and verifiable by Customer IT via network capture during IQ.

## 10. Termination

Either party may terminate **for cause** in the event of a material breach by the other party that remains uncured for 30 days after written notice. Parties may also terminate by mutual written agreement.

Upon termination:
- Licenses granted prior to termination survive **except** where termination is for Customer's material breach (e.g. unauthorized redistribution), in which case affected licenses terminate.
- Customer retains all data and reconstructions produced during the license period.
- Source escrow release conditions under [ESCROW.md](./ESCROW.md) are unaffected by this section.

## 11. Limitations and Intended Use

**Limit of Liability.** Except for Vendor's indemnity obligations under Section 8, each party's aggregate liability under this agreement is capped at the fees paid by Customer to Vendor in the 12 months preceding the claim. Neither party is liable for indirect, incidental, or consequential damages. This limitation does **not** apply to gross negligence or willful misconduct.

**Intended Use — RESEARCH ONLY.** The Product is a research tool for digital holographic microscopy. It is **not** intended, validated, or certified for:
- In-vitro diagnostic (IVD) use.
- Clinical decision-making.
- Any regulated medical device workflow under IVDR, FDA 510(k), or equivalent.

Customer shall not represent or deploy the Product in any such workflow without a separate written agreement.

## 12. Escrow

Source escrow terms — deposit materials, agent selection, release conditions, and verification cadence — are set out in full in [ESCROW.md](./ESCROW.md). Escrow agent fees are paid by Vendor.

## 13. Governing Law and Dispute Resolution

Governing law: **TBD** `[NEGOTIABLE]`. Vendor proposes **International Chamber of Commerce (ICC) Rules of Arbitration, seat in Istanbul, Turkey, proceedings in English**, with a single arbitrator. Flag: Customer jurisdiction and preferred seat to be confirmed by Customer Legal.

## 14. Signatures

| For Vendor (Hybrid Optics) | For Customer ([Customer Name]) |
|---|---|
| Name: __________________ | Name: __________________ |
| Title: __________________ | Title: __________________ |
| Date: __________________ | Date: __________________ |
| Signature: ______________ | Signature: ______________ |

Effective Date: **[Effective Date]**.
