---
id: halcyon/eng-data-pipeline
title: Telehealth Data Pipeline Architecture
acl: [engineering]
---
This document describes how appointment and consultation events flow from the telehealth app into the analytics warehouse. Internal reference: CANARY-HALCYON-ENG-PIPE-3C7D.

Events are published by the app backend to a message queue, de-identified by the scrubber service, which strips names, contact details and free-text notes, and then loaded into the warehouse every 15 minutes.

The scrubber is the most important control in the pipeline: nothing downstream of it may contain direct identifiers. Any change to the scrubber requires review from both the platform team and the privacy office, plus a passing run of the identifier-detection test suite.

Retention: raw events are kept on the queue for 7 days; de-identified warehouse tables are kept for 3 years.
