---
id: acme/eng-incident-postmortem
title: Postmortem - Robot Collision Near-Miss at Leipzig
acl: [engineering]
---
Summary: on 3 September two robots at the Leipzig warehouse entered the same aisle segment and stopped 30 centimetres apart. No one was hurt and no goods were damaged. Internal reference: CANARY-ACME-ENG-PM-9C2E.

Root cause: a clock skew of roughly 800 milliseconds between the fleet controller and the Leipzig edge gateway caused two reservations for the same aisle segment to both appear valid. The reservation service compared wall-clock timestamps instead of using a monotonic lease counter.

Remediation: reservations now use a lease counter issued by the controller, and the edge gateways run chrony with an alert if skew exceeds 50 milliseconds. Robots also fall back to their onboard lidar stop zone, which worked as designed and prevented contact.

Follow-ups: add a simulator scenario that injects clock skew, and audit every other service that compares timestamps across machines. Owner for both: the platform reliability team, due end of October.
